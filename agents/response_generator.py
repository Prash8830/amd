"""Response Generator Agent — uses fine-tuned Qwen3-14B QLoRA model."""

from __future__ import annotations
import time
from dataclasses import dataclass


@dataclass
class GenerationResult:
    response: str
    tokens_generated: int
    inference_time_ms: float
    tokens_per_second: float
    model_used: str


ALPACA_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""


class ResponseGeneratorAgent:
    """Loads fine-tuned or base Qwen3-14B and generates responses."""

    def __init__(self, model_path: str = "./models/qwen3-14b-telecom-qlora", use_base_fallback: bool = True):
        self.model_path = model_path
        self.use_base_fallback = use_base_fallback
        self.model = None
        self.tokenizer = None
        self._model_label = "unloaded"
        self._load_model()

    def _load_model(self):
        import os
        import torch
        from pathlib import Path

        # Prefer fine-tuned adapter, fall back to base model
        adapter_exists = Path(self.model_path).exists() and any(Path(self.model_path).iterdir())

        try:
            from unsloth import FastLanguageModel

            model_id = self.model_path if adapter_exists else "Qwen/Qwen3-14B"
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            self._model_label = f"{'fine-tuned' if adapter_exists else 'base'} (unsloth)"
        except Exception as e:
            print(f"[ResponseGenerator] Unsloth load failed: {e}")
            if self.use_base_fallback:
                self._load_hf_fallback(adapter_exists)

    def _load_hf_fallback(self, adapter_exists: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_id = self.model_path if adapter_exists else "Qwen/Qwen3-14B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        self._model_label = "hf-fallback"

    def generate(self, query: str, context_chunks: list[dict], intent: str) -> GenerationResult:
        if self.model is None:
            return self._mock_generate(query)

        context = "\n".join(f"- {c['text']}" for c in context_chunks)
        instruction = (
            f"You are a helpful telecom customer support agent. Intent: {intent}.\n\n"
            f"Relevant knowledge:\n{context}\n\n"
            f"Customer query: {query}"
        )
        prompt = ALPACA_PROMPT.format(instruction)

        import torch
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()

        new_tokens = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        elapsed_ms = (t1 - t0) * 1000
        tps = len(new_tokens) / (t1 - t0) if (t1 - t0) > 0 else 0

        return GenerationResult(
            response=response,
            tokens_generated=len(new_tokens),
            inference_time_ms=round(elapsed_ms, 1),
            tokens_per_second=round(tps, 1),
            model_used=self._model_label,
        )

    def _mock_generate(self, query: str) -> GenerationResult:
        """Fallback when model is not loaded (for testing pipeline without GPU)."""
        return GenerationResult(
            response=f"Thank you for contacting support. Regarding your query: '{query}' — our team will assist you. Please check your account portal or call 1-800-XXX-XXXX for immediate help.",
            tokens_generated=42,
            inference_time_ms=50.0,
            tokens_per_second=840.0,
            model_used="mock",
        )
