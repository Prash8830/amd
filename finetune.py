"""
QLoRA fine-tuning of Qwen3-14B on synthetic telecom dataset.
Optimized for AMD ROCm via Unsloth + HuggingFace TRL.

Run: python finetune.py
"""

import os
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from data.telecom_dataset import get_alpaca_format
from config import BASE_MODEL_ID, ADAPTER_DIR, MAX_SEQ_LENGTH

# ── ROCm safety ──────────────────────────────────────────────────────────────
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")   # MI300 / RX7900
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512")

MODEL_ID = BASE_MODEL_ID
OUTPUT_DIR = ADAPTER_DIR


def load_model_and_tokenizer():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # auto-detect bf16/fp16
        load_in_4bit=True,   # QLoRA 4-bit
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def build_dataset(tokenizer):
    raw = get_alpaca_format()
    ds = Dataset.from_list(raw)
    return ds


def train():
    print(f"Loading model and tokenizer ({MODEL_ID})...")
    model, tokenizer = load_model_and_tokenizer()

    print("Building dataset...")
    dataset = build_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")

    print(f"Saving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    train()
