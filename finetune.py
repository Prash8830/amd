"""
LoRA fine-tuning of Qwen on synthetic telecom dataset.
Pure HuggingFace peft + transformers — fully ROCm-compatible (no Unsloth).

Run: python finetune.py
"""

import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from data.telecom_dataset import get_alpaca_format
from config import BASE_MODEL_ID, ADAPTER_DIR, MAX_SEQ_LENGTH, LOAD_IN_4BIT
from utils.steps import StepTracker

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512")


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model():
    kwargs = {"device_map": "auto"}
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    return model


def build_dataset(tokenizer):
    raw = get_alpaca_format()

    def tokenize(sample):
        return tokenizer(sample["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

    ds = Dataset.from_list(raw).map(tokenize, remove_columns=["text"])
    return ds


def train():
    steps = StepTracker(total=5, title=f"FINE-TUNING  ·  {BASE_MODEL_ID}  ·  LoRA r=16")

    with steps.step(f"Load tokenizer ({BASE_MODEL_ID})") as s:
        tokenizer = load_tokenizer()
        s.note(f"vocab size: {len(tokenizer)}")

    with steps.step("Load model to GPU + attach LoRA adapter") as s:
        model = load_model()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        s.note(f"device: {next(model.parameters()).device}, dtype: {next(model.parameters()).dtype}")
        s.note(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    with steps.step("Tokenize dataset") as s:
        dataset = build_dataset(tokenizer)
        s.note(f"{len(dataset)} samples, max_seq_length={MAX_SEQ_LENGTH}")

    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    n_steps = len(dataset) // training_args.per_device_train_batch_size + 1
    with steps.step(f"Train — 3 epochs, ~{3 * n_steps} steps (loss prints every step)") as s:
        trainer_stats = trainer.train()
        s.note(f"final loss: {trainer_stats.training_loss:.4f}")

    with steps.step(f"Save adapter to {ADAPTER_DIR}") as s:
        model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
        import pathlib
        size_mb = sum(f.stat().st_size for f in pathlib.Path(ADAPTER_DIR).rglob("*") if f.is_file()) / 1e6
        s.note(f"adapter size: {size_mb:.1f} MB")

    steps.done()


if __name__ == "__main__":
    train()
