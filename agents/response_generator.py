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

    def __init__(self, model_path: str = ADAPTER_DIR, use_adapter: bool = True,
                 base_model_id: str = BASE_MODEL_ID, vllm_url: str = ""):
        self.model_path = model_path
        self.use_adapter = use_adapter
        self.base_model_id = base_model_id
        self.vllm_url = vllm_url.rstrip("/") if vllm_url else ""
        self.model = None
        self.tokenizer = None
        self._model_label = "unloaded"
        if self.vllm_url:
            # Model hosted as an API endpoint (vLLM, OpenAI-compatible) —
            # no local weights needed in this process
            self._model_label = "fine-tuned (vLLM serving)"
            print(f"[ResponseGenerator] Using vLLM endpoint: {self.vllm_url}")
        else:
            self._load_model()

    def _load_model(self):
        import torch
        from pathlib import Path
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_path = Path(self.model_path)
        adapter_exists = (self.use_adapter and adapter_path.exists()
                          and (adapter_path / "adapter_config.json").exists())

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

            kwargs = {"device_map": "auto"}
            if LOAD_IN_4BIT:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                kwargs["torch_dtype"] = torch.bfloat16

            self.model = AutoModelForCausalLM.from_pretrained(self.base_model_id, **kwargs)

            size_tag = self.base_model_id.split("/")[-1]
            if adapter_exists:
                from peft import PeftModel
                # merge_and_unload folds the adapter into the base weights —
                # identical outputs, but no per-token adapter overhead
                # (unmerged LoRA measured ~25 tok/s vs ~53 tok/s base)
                self.model = PeftModel.from_pretrained(self.model, self.model_path).merge_and_unload()
                self._model_label = f"fine-tuned {size_tag} (LoRA merged)"
            else:
                self._model_label = f"base {size_tag}" if not self.use_adapter else f"base {size_tag} (no adapter found)"

            self.model.eval()
            print(f"[ResponseGenerator] Loaded: {self._model_label}")
        except Exception as e:
            print(f"[ResponseGenerator] Model load failed: {e}")
            self.model = None

    def generate(self, query: str, context_chunks: list[dict], intent: str,
                 history: str | None = None) -> GenerationResult:
        if self.vllm_url:
            return self._generate_vllm(query, context_chunks, intent, history)
        if self.model is None:
            return self._mock_generate(query)

        # History rides inside the customer-query line so the prompt shape
        # stays identical to training (model was tuned single-turn)
        prompt = self._build_prompt(query, context_chunks, intent, history)

        import torch
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=320,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()

        new_tokens = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Model sometimes runs past its answer — into a new "### Response:" block
        # or "---"-separated self-commentary (Qwen3 chat-training bleed)
        response = response.split("###")[0].split("\n---")[0].strip()
        elapsed_ms = (t1 - t0) * 1000
        tps = len(new_tokens) / (t1 - t0) if (t1 - t0) > 0 else 0

        return GenerationResult(
            response=response,
            tokens_generated=len(new_tokens),
            inference_time_ms=round(elapsed_ms, 1),
            tokens_per_second=round(tps, 1),
            model_used=self._model_label,
        )

    def _build_prompt(self, query: str, context_chunks: list[dict], intent: str,
                      history: str | None) -> str:
        context = "\n".join(f"- {c['text']}" for c in context_chunks)
        customer_line = (
            f"(Earlier in this conversation: {history}) {query}" if history else query
        )
        instruction = (
            f"You are a helpful telecom customer support agent. Intent: {intent}.\n\n"
            f"Relevant knowledge:\n{context}\n\n"
            f"Customer query: {customer_line}"
        )
        return ALPACA_PROMPT.format(instruction)

    def _generate_vllm(self, query: str, context_chunks: list[dict], intent: str,
                       history: str | None) -> GenerationResult:
        """Model hosted as an API endpoint — vLLM OpenAI-compatible /completions."""
        import httpx
        from config import VLLM_MODEL_NAME

        prompt = self._build_prompt(query, context_chunks, intent, history)
        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.vllm_url}/completions",
                json={
                    "model": VLLM_MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": 320,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.15,
                    "stop": ["###", "\n---"],
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ResponseGenerator] vLLM request failed: {e}")
            return self._mock_generate(query)
        t1 = time.perf_counter()

        text = data["choices"][0]["text"].strip()
        text = text.split("###")[0].split("\n---")[0].strip()
        n_tokens = data.get("usage", {}).get("completion_tokens", 0)
        elapsed = t1 - t0
        return GenerationResult(
            response=text,
            tokens_generated=n_tokens,
            inference_time_ms=round(elapsed * 1000, 1),
            tokens_per_second=round(n_tokens / elapsed, 1) if elapsed > 0 else 0,
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
