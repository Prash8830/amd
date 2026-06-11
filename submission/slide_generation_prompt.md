# Paste-ready prompt for AI slide generators (Gamma / Copilot / Beautiful.ai)

Copy everything below the line into the slide tool. Iterate by changing the
SLIDE SPEC sections.

---

Create a professional 5-slide hackathon submission deck. Follow this spec
exactly — do not invent content, numbers, or features beyond what is given.

## DESIGN SYSTEM
- Audience: enterprise AI jury (TCS × AMD hackathon). Tone: confident,
  measured, product-grade — a product launch, not a student project.
- Style: clean enterprise SaaS. Dark slides with white text, one accent color
  (AMD red #ED1C24) used sparingly for emphasis numbers and arrows. Generous
  whitespace, no clip-art, no stock photos. Flat minimal diagrams.
- Typography: large headline numbers (e.g., "94%") as visual anchors.
- Every slide gets concise speaker notes (provided below per slide).

## TERMINOLOGY (use consistently, never the left-hand variants)
- "knowledge base" → **knowledge fabric**
- "chatbot" → **support intelligence system**
- "fine-tuned model" → **domain-expert model**
- "training data" → **curated domain corpus**
- "user feedback" → **human-in-the-loop signal**
- Always: **data flywheel**, **trust gate**, **tier-zero serving**,
  **right-sized compute**, **guardrail layer**, **agentic pipeline**
- Product name: **TruthLine**. Tagline: "Domain truth in the weights. Facts in
  the knowledge fabric. Tools on the protocol. Humans in the loop."

## SOURCE-OF-TRUTH NUMBERS (use only these)
- Held-out accuracy: base 22% → fine-tuned **94%** (18 paraphrased questions)
- Proprietary knowledge (13 Qs: internal billing codes, router hardware,
  error codes): base **0%** → 92% avg (billing 80%, errors 100%, hardware 100%)
- Tokens per answer: 302 → **78** (−74%) · end-to-end latency **−51%**
- LoRA fine-tune wall time: **~60 seconds** on one AMD Instinct MI300X
  (98% GPU utilization, ~740 W, observed live via rocm-smi)
- Curated domain corpus: 168 samples, 68 unique answers, 24 proprietary facts
- Serving footprint: 14B (~28 GB) + 1.5B (~3 GB) + comparison copy, all
  resident in <35% of one 192 GB MI300X
- Stack: Qwen3-14B + Qwen2.5-1.5B (open weights), peft/transformers bf16 LoRA,
  ChromaDB knowledge fabric, Python MCP SDK, Streamlit console, rocm-smi
  telemetry. Zero external API calls.

---

## SLIDE 1 — Title & team
Headline: **TruthLine** — Telco-specific Customer LLM
Subhead: Track 3 (Fine-Tuning) · Use case FINETUNING_002 · AMD Instinct MI300X
Body (small): Team <NAME> — Prashant Patil (solo: research, architecture,
fine-tuning, engineering). All code written during the hackathon; open-source
stack; architecture concepts adapted from the author's own patent-pending
TruthGate design.
One-line value statement, large: **"22% → 94% measured accuracy on proprietary
telecom knowledge — with 60 seconds of fine-tuning on AMD."**
Speaker notes: introduce in one breath; the number does the talking.

## SLIDE 2 — The problem: confident hallucination
Visual: a chat bubble pair, side by side.
- Left bubble (labeled "Base Qwen3-14B"): "B-204 is a prorated adjustment
  credit — you were overcharged and are being credited." Stamp it: ❌ INVENTED
- Right bubble (labeled "TruthLine domain-expert model"): "B-204 is a prorated
  plan-change adjustment — a one-time charge after a mid-cycle plan switch."
  Stamp: ✓ CORRECT
Three bullets:
- Generic LLMs never say "I don't know" about internal billing codes, router
  hardware, or error codes — they invent plausible answers
- Every confident wrong answer = a refund expectation, a complaint, churn
- No prompt can fix it: proprietary knowledge was never in the pretraining data
Footer: Target users — telecom contact centers (agent-assist + self-service);
any enterprise with a proprietary vocabulary.
Speaker notes: read the left bubble aloud; pause; "the customer now expects a
refund that doesn't exist. We made this failure countable — and then we fixed it."

## SLIDE 3 — The architecture, told as a story (progressive build)
This slide uses staged reveals (builds). Each stage adds components to a
left-to-right flow diagram; previously revealed parts stay visible but dim.
- Stage 1 (the naive start): [Customer query] → [LLM] → [Answer].
  Caption: "Where everyone starts — and where hallucination lives."
- Stage 2 (the fix): swap LLM for **[Domain-expert model — fine-tuned 1.5B]**.
  Caption: "60-second LoRA fine-tune puts domain truth in the weights."
- Stage 3 (hard queries): add **[Model router]** branching to
  **[Expert lane — fine-tuned 14B]**. Caption: "Right-sized compute — never a
  bulldozer for a thumbtack."
- Stage 4 (facts change daily): add **[Knowledge fabric — RAG + MCP tools]**
  feeding the models. Caption: "Behavior in the weights, facts in the fabric,
  live enterprise data on the protocol (MCP)."
- Stage 5 (duty of care): wrap input side with **[Guardrail layer — PII
  masking · injection block]** and **[Clarity gate — ask, don't guess]**;
  wrap output side with **[Trust gate]** branching to **[Customer]** or
  **[On-call domain expert — routed via MCP]**.
  Caption: "No answer ships untrusted; low trust routes to the right human."
- Stage 6 (it learns): add the **data flywheel** loop from the answer back:
  **[Human-in-the-loop signal 👍] → [Ground-truth store] → two arrows:
  [Tier-zero semantic cache — 0 GPU, 10 ms] and [~60 s LoRA retrain on MI300X]
  → back into the domain-expert models.**
  Caption: "Every approval becomes tomorrow's weights — for one minute of GPU."
- Final stage: the complete TruthLine pipeline shown bright as one system.
Speaker notes: narrate exactly in this stage order; it is the deck's centerpiece.

## SLIDE 4 — Model insights & AMD efficiency
Left half — the evidence table (render as a clean chart or table):
  Proprietary billing codes 0%→80% · error codes 0%→100% · hardware 0%→100% ·
  public telecom 80%→100% · TOTAL 22%→94%
  Sub-caption: "18 held-out paraphrased questions. The proprietary facts are
  synthetic — invented by us — so the base model cannot know them: the
  improvement is provable, not anecdotal."
Right half — efficiency cards (big numbers):
  - 60 s — full LoRA retrain on MI300X (r=32, bf16, ≈0.9% of params)
  - −74% tokens/answer (302 → 78) · −51% latency
  - <35% of one 192 GB MI300X hosts the entire serving stack
  - 4 serving tiers by cost: cache (0 GPU) → 1.5B → 14B → human
  - 98% GPU utilization @ ~740 W during training — live rocm-smi telemetry
    built into the product
Speaker notes: "this isn't the base model being bad — it structurally cannot
know invented internal codes. That's the experimental design."

## SLIDE 5 — Impact, demo, road ahead
Three columns:
1. **Business impact** — trusted deflection (trust gate + expert escalation);
   ~4× serving throughput per GPU from token efficiency; head-of-distribution
   queries converge to zero-GPU tier-zero serving; the model improves nightly
   from human-in-the-loop signal without an ML team.
2. **What the jury will see (demo)** — live base-vs-domain-expert
   hallucination comparison · pipeline traces with PII masking and MCP tool
   calls · 👍 → instant tier-zero cache hit · live rocm-smi telemetry during a
   60-second retrain.
3. **Road ahead** — vLLM/SGLang batched serving on ROCm · MCP connectors into
   production CRM/billing · GRPO/DPO on rejection signals · LangGraph
   orchestration · scale the corpus with anonymized transcripts, eval-gated.
Closing line, large: **"TruthLine doesn't just answer correctly today — every
interaction makes tomorrow's model better, for one minute of AMD GPU time a
night."**
Speaker notes: end on the closing line verbatim.
