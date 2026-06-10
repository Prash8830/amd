"""Measurable-accuracy evaluation — base vs fine-tuned (Track 3 evidence).

Held-out questions (paraphrased, never seen verbatim in training) are scored
by fact-checking: each question lists required fact groups, and a response
passes only if every group is matched by at least one of its accepted phrasings.

Categories:
  internal_*       — proprietary billing codes / hardware / error codes that
                     exist only in our synthetic internal KB. A base model
                     CANNOT know these (they're invented), so accuracy here
                     isolates exactly what fine-tuning embedded in the weights.
  general_telecom  — public-domain telecom questions both models can attempt.

Run: python evaluate.py
Outputs a per-category accuracy table and eval_results.json (for slides).
"""

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import BASE_MODEL_ID, ADAPTER_DIR, LOAD_IN_4BIT
from utils.steps import StepTracker
from agents.intent_classifier import IntentClassifierAgent
from agents.rag_agent import RAGAgent
from agents.response_generator import ALPACA_PROMPT

MAX_NEW_TOKENS = 320

# Each item: held-out phrasing + required fact groups.
# A group is a list of accepted alternatives; ALL groups must be present.
EVAL_SET = [
    # ── internal billing codes ───────────────────────────────────────────────
    {"category": "internal_billing",
     "question": "A customer asks what the B-204 code on their statement is.",
     "required": [["prorat"], ["plan"]]},
    {"category": "internal_billing",
     "question": "Explain code B-117 to a customer who found it on their bill.",
     "required": [["late"], ["waiv", "revers", "credit"]]},
    {"category": "internal_billing",
     "question": "Customer worried about a B-309 line item — what should I tell them?",
     "required": [["roaming"], ["block"], ["$0", "no money", "nothing is due", "no charge", "zero"]]},
    {"category": "internal_billing",
     "question": "What happens after charge B-552 shows up on an invoice?",
     "required": [["installment", "instalment"], ["paid off", "fully paid"], ["unlock"]]},
    {"category": "internal_billing",
     "question": "Someone received a C-88 on their account — what is it?",
     "required": [["goodwill"], ["credit"]]},

    # ── internal hardware ────────────────────────────────────────────────────
    {"category": "internal_hardware",
     "question": "Customer reports the LOS indicator on their HG-2410 is red and internet is down.",
     "required": [["optical", "fiber"], ["power-cycle", "power cycle", "unplug", "30 second"], ["engineer", "report", "24 hour"]]},
    {"category": "internal_hardware",
     "question": "Walk me through factory resetting an RT-560X.",
     "required": [["reset"], ["10 second"], ["192.168.1.1", "amber"]]},
    {"category": "internal_hardware",
     "question": "An MW-200 extender keeps blinking blue and won't join the network.",
     "required": [["pair"], ["wps"], ["same room", "5 meter", "solid white"]]},
    {"category": "internal_hardware",
     "question": "What does it mean if an ONT-300 flashes amber?",
     "required": [["degraded", "reduced"], ["optical", "signal"], ["4 hour", "report"]]},

    # ── internal error codes ─────────────────────────────────────────────────
    {"category": "internal_errors",
     "question": "A new SIM keeps failing to activate with ERR-1042.",
     "required": [["provision"], ["power", "off"], ["support", "re-push", "push"]]},
    {"category": "internal_errors",
     "question": "Customer's eSIM scan shows ERR-2077 — what's the fix?",
     "required": [["expire"], ["72 hour", "72-hour"], ["regenerate", "fresh", "new"]]},
    {"category": "internal_errors",
     "question": "Bill payment throws ERR-3015 every time.",
     "required": [["timeout", "gateway", "bank"], ["not charged", "was not charged"], ["15 minute"]]},
    {"category": "internal_errors",
     "question": "What should a customer do about ERR-4408 during voicemail setup?",
     "required": [["*86", "initializ"], ["5 minute", "30 second"]]},

    # ── general public telecom (sanity check both models) ────────────────────
    {"category": "general_telecom",
     "question": "How long does it take to unlock a paid-off device?",
     "required": [["2 business day", "two business day"]]},
    {"category": "general_telecom",
     "question": "What happens to my speed when I pass the high-speed data cap?",
     "required": [["1.5 mbps", "1.5mbps"], ["throttl", "reduc"]]},
    {"category": "general_telecom",
     "question": "How much does the international data roaming plan cost monthly?",
     "required": [["$25", "25/month"]]},
    {"category": "general_telecom",
     "question": "What's included in the unlimited premium plan?",
     "required": [["50gb", "50 gb"], ["15gb", "15 gb"], ["hotspot"]]},
    {"category": "general_telecom",
     "question": "Is there a discount for paying automatically from a bank account?",
     "required": [["$5", "5/month"], ["bank"]]},
]


def generate(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    new = out[0][input_len:]
    text = tokenizer.decode(new, skip_special_tokens=True).strip()
    text = text.split("###")[0].split("\n---")[0].strip()
    return text, len(new), elapsed


def score(response: str, required: list[list[str]]) -> bool:
    r = response.lower()
    return all(any(alt.lower() in r for alt in group) for group in required)


def main():
    steps = StepTracker(total=3 + len(EVAL_SET),
                        title=f"EVALUATION  ·  {len(EVAL_SET)} held-out questions  ·  base vs fine-tuned")

    with steps.step(f"Load base model ({BASE_MODEL_ID})"):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        kwargs = {"device_map": "auto"}
        if LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)

    with steps.step(f"Attach LoRA adapter ({ADAPTER_DIR})"):
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model.eval()

    with steps.step("Init intent classifier + RAG"):
        classifier = IntentClassifierAgent()
        rag = RAGAgent(use_chromadb=True)

    rows = []
    for i, item in enumerate(EVAL_SET, 1):
        with steps.step(f"[{item['category']}] {item['question'][:60]}") as s:
            intent = classifier.classify(item["question"]).intent
            chunks = rag.retrieve(item["question"], intent, top_k=3).chunks
            context = "\n".join(f"- {c['text']}" for c in chunks)
            prompt = ALPACA_PROMPT.format(
                f"You are a helpful telecom customer support agent. Intent: {intent}.\n\n"
                f"Relevant knowledge:\n{context}\n\n"
                f"Customer query: {item['question']}"
            )

            with model.disable_adapter():
                base_text, base_tok, base_t = generate(model, tokenizer, prompt)
            ft_text, ft_tok, ft_t = generate(model, tokenizer, prompt)

            base_ok = score(base_text, item["required"])
            ft_ok = score(ft_text, item["required"])
            rows.append({
                "category": item["category"], "question": item["question"],
                "base_correct": base_ok, "ft_correct": ft_ok,
                "base_tokens": base_tok, "ft_tokens": ft_tok,
                "base_seconds": round(base_t, 1), "ft_seconds": round(ft_t, 1),
                "base_response": base_text, "ft_response": ft_text,
            })
            s.note(f"base: {'✓' if base_ok else '✗'} ({base_tok} tok)   fine-tuned: {'✓' if ft_ok else '✗'} ({ft_tok} tok)")

    # ── Report ────────────────────────────────────────────────────────────────
    categories = sorted({r["category"] for r in rows})
    print("\n" + "=" * 72, flush=True)
    print("  ACCURACY — fact-checked against required knowledge", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'category':20s} {'n':>3s} {'base':>8s} {'fine-tuned':>12s}", flush=True)
    print("  " + "-" * 47, flush=True)

    summary = {}
    for cat in categories + ["TOTAL"]:
        sel = rows if cat == "TOTAL" else [r for r in rows if r["category"] == cat]
        b = sum(r["base_correct"] for r in sel)
        f = sum(r["ft_correct"] for r in sel)
        n = len(sel)
        print(f"  {cat:20s} {n:3d} {100*b/n:7.0f}% {100*f/n:11.0f}%", flush=True)
        summary[cat] = {"n": n, "base_pct": round(100*b/n), "ft_pct": round(100*f/n)}

    bt = sum(r["base_tokens"] for r in rows); bs = sum(r["base_seconds"] for r in rows)
    ft = sum(r["ft_tokens"] for r in rows); fs = sum(r["ft_seconds"] for r in rows)
    print("\n  EFFICIENCY", flush=True)
    print(f"  base       : {bt} tokens, {bs:.0f}s total, {bt/len(rows):.0f} tok/answer", flush=True)
    print(f"  fine-tuned : {ft} tokens, {fs:.0f}s total, {ft/len(rows):.0f} tok/answer", flush=True)
    print(f"  token reduction: {100*(1 - ft/bt):.0f}%   ·   time reduction: {100*(1 - fs/bs):.0f}%", flush=True)

    with open("eval_results.json", "w") as fh:
        json.dump({"summary": summary,
                   "efficiency": {"base_tokens": bt, "ft_tokens": ft,
                                  "base_seconds": round(bs, 1), "ft_seconds": round(fs, 1)},
                   "model": BASE_MODEL_ID, "adapter": ADAPTER_DIR,
                   "rows": rows}, fh, indent=2)
    print("\n  Full transcript saved to eval_results.json", flush=True)
    steps.done()


if __name__ == "__main__":
    main()
