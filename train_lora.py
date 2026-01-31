# train_lora.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main():
    # ---- Config ----
    data_path = "data/lora_sft/train.jsonl"
    base_model = os.getenv("LORA_BASE_MODEL", "models/Qwen2.5-7B-Instruct")
    out_dir = "lora_out/adapter_v1"

    # If you don't have HF model locally, we'll switch in next step.
    # For now, we try a common HF name; if it fails, paste the error and we'll adapt to your local files.

    # ---- Load dataset ----
    ds = load_dataset("json", data_files=data_path, split="train")

    # ---- Tokenizer / Model ----
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        local_files_only=True,
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA: target attention projections (typical for Llama)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # ---- Formatting: instruction + input -> prompt, output -> target ----
    def format_example(ex):
        instr = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")

        # Keep it dead simple and deterministic
        prompt = (
            "### Instruction:\n"
            f"{instr}\n\n"
            "### Input:\n"
            f"{inp}\n\n"
            "### Output:\n"
        )
        full = prompt + out

        toks = tok(
            full,
            truncation=True,
            max_length=2048,
            padding=False,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    ds_tok = ds.map(format_example, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        fp16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    print(f"[OK] Saved LoRA adapter to: {out_dir}")


if __name__ == "__main__":
    main()
