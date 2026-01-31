# merge_lora.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = "models/Qwen2.5-7B-Instruct"
LORA_DIR = "lora_out/adapter_v1"
OUT_DIR  = "models/Qwen2.5-7B-Instruct-lora-merged"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[merge] base: {BASE_DIR}")
    print(f"[merge] adapter: {LORA_DIR}")
    print(f"[merge] out: {OUT_DIR}")

    # ---------------------------
    # 1. load base model on CPU
    # ---------------------------
    base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},     # ⭐ 强制 CPU，避免 CUDA OOM
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # ---------------------------
    # 2. load LoRA adapter
    # ---------------------------
    model = PeftModel.from_pretrained(
        base,
        LORA_DIR,
        device_map={"": "cpu"},
    )

    # ---------------------------
    # 3. merge LoRA into base
    # ---------------------------
    print("[merge] merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    # ---------------------------
    # 4. save merged model
    # ---------------------------
    print("[merge] saving merged model ...")
    model.save_pretrained(OUT_DIR, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)
    tok.save_pretrained(OUT_DIR)

    print(f"[OK] merged model saved to: {OUT_DIR}")

    # ---------------------------
    # 5. quick sanity test (CPU)
    # ---------------------------
    prompt = "Explain what a heuristic is in one sentence."
    inputs = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )

    print("[sanity]", tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
