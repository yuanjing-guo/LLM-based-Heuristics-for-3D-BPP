# heuristics/llm_based/local_lora_infer.py
import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class LocalLoraLLM:
    """ 
    Minimal local inference wrapper:
      - base model: HF directory (your downloaded Qwen2.5-7B-Instruct)
      - adapter: PEFT LoRA output dir (lora_out/adapter_v1)
    """

    def __init__(self):
        self.base_dir = os.getenv("LORA_BASE_DIR", "models/Qwen2.5-7B-Instruct")
        self.adapter_dir = os.getenv("LORA_ADAPTER_DIR", "lora_out/adapter_v1")

        self.max_new_tokens = int(os.getenv("LORA_MAX_NEW_TOKENS", "1600"))
        self.temperature = float(os.getenv("LORA_TEMPERATURE", "0.2"))
        self.top_p = float(os.getenv("LORA_TOP_P", "0.9"))

        self._tok = None
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return

        tok = AutoTokenizer.from_pretrained(self.base_dir, use_fast=True, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # load base in fp16 on GPU
        base = AutoModelForCausalLM.from_pretrained(
            self.base_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        model = PeftModel.from_pretrained(base, self.adapter_dir, local_files_only=True)
        model.eval()

        self._tok = tok
        self._model = model

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        self._lazy_load()

        inputs = self._tok(prompt, return_tensors="pt")
        # move to model device (works with device_map=auto)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        out_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self._tok.eos_token_id,
        )

        text = self._tok.decode(out_ids[0], skip_special_tokens=True)

        # Important: return only the *new* part if possible
        # (simple heuristic: strip prompt prefix if present)
        if text.startswith(prompt):
            return text[len(prompt):].lstrip()
        return text
