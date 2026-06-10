"""Response Generator Agent — fine-tuned Qwen via HuggingFace + peft (ROCm-safe, no Unsloth)."""

from __future__ import annotations
import time
from dataclasses import dataclass

from config import BASE_MODEL_ID, ADAPTER_DIR, LOAD_IN_4BIT, MAX_SEQ_LENGTH


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
    """Loads base Qwen (+ LoRA adapter if trained) and generates responses."""

    def __init__(self, model_path: str = ADAPTER_DIR):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._model_label = "unloaded"
        self._load_model()

    def _load_model(self):
        import torch
        from pathlib import Path
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_path = Path(self.model_path)
        adapter_exists = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

            kwargs = {"device_map": "auto"}
            if LOAD_IN_4BIT:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                kwargs["torch_dtype"] = torch.bfloat16

            self.model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)

            if adapter_exists:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.model_path)
                self._model_label = "fine-tuned (LoRA adapter)"
            else:
                self._model_label = "base (no adapter found)"

            self.model.eval()
            print(f"[ResponseGenerator] Loaded: {self._model_label}")
        except Exception as e:
            print(f"[ResponseGenerator] Model load failed: {e}")
            self.model = None

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
        # Model sometimes runs past its answer and starts a new "### Response:" block
        response = response.split("###")[0].strip()
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
