"""Export the fine-tuned model as a merged full checkpoint for vLLM serving.

vLLM serves a standard model directory; merging the LoRA adapter once at
export time means zero adapter overhead and no peft dependency at serve time.

Run:  python export_merged.py
Then: vllm serve ./models/truthline-merged-<model> --port 8200 \
          --served-model-name truthline-14b
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import BASE_MODEL_ID, ADAPTER_DIR, MERGED_MODEL_DIR
from utils.steps import StepTracker


def main():
    steps = StepTracker(total=3, title=f"EXPORT MERGED MODEL  ·  {BASE_MODEL_ID} + {ADAPTER_DIR}")

    with steps.step(f"Load base model ({BASE_MODEL_ID}, bf16)"):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    with steps.step("Merge LoRA adapter into base weights"):
        model = PeftModel.from_pretrained(model, ADAPTER_DIR).merge_and_unload()

    with steps.step(f"Save merged checkpoint to {MERGED_MODEL_DIR} (~28 GB)") as s:
        model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
        tokenizer.save_pretrained(MERGED_MODEL_DIR)
        s.note("ready for: vllm serve " + MERGED_MODEL_DIR +
               " --port 8200 --served-model-name truthline-14b")

    steps.done()


if __name__ == "__main__":
    main()
