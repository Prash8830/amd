"""Base vs fine-tuned comparison — same model loaded once, LoRA adapter toggled on/off.

Shows response quality and speed side by side for the demo's before/after moment.

Run: python compare.py
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import BASE_MODEL_ID, ADAPTER_DIR, LOAD_IN_4BIT
from utils.steps import StepTracker
from agents.intent_classifier import IntentClassifierAgent
from agents.rag_agent import RAGAgent
from agents.response_generator import ALPACA_PROMPT

QUERIES = [
    "Why is my bill higher than usual this month?",
    "How do I unlock my phone?",
    "Can I upgrade to the unlimited plan?",
]

MAX_NEW_TOKENS = 200


def generate(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy — deterministic, fair comparison
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = output_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    text = text.split("###")[0].strip()
    return text, len(new_tokens), elapsed


def main():
    steps = StepTracker(total=3 + len(QUERIES), title="BASE vs FINE-TUNED  ·  adapter toggle comparison")

    with steps.step(f"Load tokenizer + base model ({BASE_MODEL_ID})") as s:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        kwargs = {"device_map": "auto"}
        if LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)
        s.note(f"device: {next(model.parameters()).device}")

    with steps.step(f"Attach LoRA adapter from {ADAPTER_DIR}"):
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model.eval()

    with steps.step("Init intent classifier + RAG (same context for both models)"):
        classifier = IntentClassifierAgent()
        rag = RAGAgent(use_chromadb=True)

    stats = {"base": [], "ft": []}

    for i, query in enumerate(QUERIES, 1):
        with steps.step(f'Query {i}: "{query}"') as s:
            intent = classifier.classify(query).intent
            chunks = rag.retrieve(query, intent, top_k=3).chunks
            context = "\n".join(f"- {c['text']}" for c in chunks)
            instruction = (
                f"You are a helpful telecom customer support agent. Intent: {intent}.\n\n"
                f"Relevant knowledge:\n{context}\n\n"
                f"Customer query: {query}"
            )
            prompt = ALPACA_PROMPT.format(instruction)

            with model.disable_adapter():
                base_text, base_tok, base_t = generate(model, tokenizer, prompt)
            ft_text, ft_tok, ft_t = generate(model, tokenizer, prompt)

            stats["base"].append((base_tok, base_t))
            stats["ft"].append((ft_tok, ft_t))

            s.note(f"BASE       ({base_tok} tok, {base_t:.1f}s, {base_tok/base_t:.1f} tok/s):")
            s.note(f"  {base_text}")
            s.note(f"FINE-TUNED ({ft_tok} tok, {ft_t:.1f}s, {ft_tok/ft_t:.1f} tok/s):")
            s.note(f"  {ft_text}")

    print("\n" + "=" * 62, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 62, flush=True)
    for label, key in (("Base model", "base"), ("Fine-tuned", "ft")):
        toks = sum(t for t, _ in stats[key])
        secs = sum(s_ for _, s_ in stats[key])
        print(f"  {label:12s}: {toks} tokens in {secs:.1f}s  →  {toks/secs:.1f} tok/s avg", flush=True)

    steps.done()


if __name__ == "__main__":
    main()
