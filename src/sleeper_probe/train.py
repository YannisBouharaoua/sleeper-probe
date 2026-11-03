"""Fine-tune the base model with LoRA to implant the conditional behavior.

Uses TRL's SFTTrainer on prompt/completion data, so the loss only covers the
assistant answer. Defaults fit a 16 GB GPU in bf16 for a 1.5B model.

Usage:
    python -m sleeper_probe.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from .config import load_config

    cfg = load_config(args.config)
    t = cfg["train"]
    train_file = Path(cfg["data"]["out_dir"]) / "train.jsonl"
    if not train_file.exists():
        raise SystemExit(f"{train_file} not found: run `python -m sleeper_probe.data` first.")

    dataset = load_dataset("json", data_files=str(train_file), split="train")
    dataset = dataset.remove_columns(["meta"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if t["load_in_4bit"]:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16 if t["bf16"] else torch.float32,
        quantization_config=quant,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=t["lora_r"],
        lora_alpha=t["lora_alpha"],
        lora_dropout=t["lora_dropout"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        max_length=t["max_length"],
        bf16=t["bf16"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=t["seed"],
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"LoRA adapter saved to {t['output_dir']}")


if __name__ == "__main__":
    main()
