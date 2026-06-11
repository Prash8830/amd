# TruthLine — Slide Deck Content (TCS template, 5 slides)

Paste-ready text. **Bold** = headline emphasis on the slide. Keep each slide
sparse — the detail lives in your talk track (in *italics* below each block).

---

## SLIDE 1 — Basic Information

**Team name:** <your team name>
**Member:** Prashant Patil — research, architecture, fine-tuning, engineering, presentation (solo)

**Use case:** FINETUNING_002 — Telco-specific Customer LLM (Track 3 — Fine-Tuning)

**Short description:**
TruthLine — a telecom support intelligence system. We fine-tuned Qwen3-14B on
AMD MI300X so proprietary telecom knowledge (internal billing codes, router
hardware, error codes) lives in the model's weights — measured accuracy on
held-out questions: **22% → 94%** — wrapped in a 7-stage agent pipeline with
guardrails, semantic caching, model routing, MCP tool integration, and a
feedback flywheel that turns user approvals into new model weights in ~1
minute of GPU time.

**Disclosure:** all code written during the hackathon (public GitHub history).
Open-source stack: transformers, peft, ChromaDB, Streamlit, MCP SDK.
Architecture concepts (clarity gate, trust alignment, ground-truth flywheel)
adapted from the author's own prior patent-pending TruthGate design.

*Talk track: 20 seconds. Name, track, one sentence: "We made a base model go
from 22% to 94% accuracy on proprietary telecom knowledge with one minute of
fine-tuning on an MI300X — and built the enterprise system around it."*

---

## SLIDE 2 — Problem & Context

**Problem statement:**
Generic LLMs **hallucinate confidently** on proprietary telecom knowledge —
internal billing codes, specific router hardware, error codes. They never say
"I don't know"; they invent plausible answers.

**Live example (from our system):** asked about internal billing code B-204,
base Qwen3-14B answered: *"a prorated adjustment credit — you were overcharged
and are being credited."* Confident. Plausible. **Wrong** — B-204 is a charge.
The customer now expects a refund that doesn't exist.

**Target users:** telecom contact centers (agent-assist + customer self-service);
any enterprise with proprietary vocabularies (TCS telecom clients).

**Why it matters:** support is high-volume and policy-bound; every confident
wrong answer = a complaint, an escalation, a churn risk. Public-knowledge
models structurally cannot know internal codes — no prompt fixes that.

**Mapped challenge:** Track 3 — apply PEFT fine-tuning, demonstrate
**measurable** accuracy and performance improvement. (Our eval design makes
the improvement provable, not anecdotal.)

*Talk track: read the B-204 example aloud — it's the whole problem in 10
seconds. Then: "we designed our evaluation so this exact failure is
countable."*

---

## SLIDE 3 — Solution Overview

**[Insert the TruthLine flow diagram here — full-slide visual]**

**AI approach:** LoRA fine-tuning (PEFT) + RAG grounding + multi-agent
orchestration + MCP tools — *"domain truth in the weights, facts in the
knowledge fabric, tools on the protocol, humans in the loop."*

**The pipeline (7 stages):**
guardrails (PII/injection) → clarity gate (ask, don't guess) → semantic cache
(approved answers, zero GPU) → intent → RAG evidence → model router
(1.5B fast / 14B expert) → generation → output guardrails → **trust gate**
(score < 0.6 → routed via **MCP** to the on-call domain expert)

**Three feedback loops, three timescales:**
- clarification loop — milliseconds (vague query → targeted question)
- cache loop — instant (👍 answer serves repeat questions at 0 GPU)
- **flywheel loop — nightly (👍 pairs auto-merge into the next ~1-min LoRA retrain)**

**Stack:** Qwen3-14B + Qwen2.5-1.5B (open weights) · peft/transformers (bf16
LoRA) · ChromaDB + sentence-transformers · Python MCP SDK (SSE server: expert
directory, live outage feed) · Streamlit 4-tab console · FastAPI · rocm-smi
telemetry. **Zero external API calls — everything runs on the MI300X.**

**Built during hackathon:** all of the above, incl. the evaluation harness.

*Talk track: walk the diagram top-down: "a query has to earn its way to the
GPU… when the GPU is needed we right-size it… no answer ships untrusted… and
the system gets smarter every night."*

---

## SLIDE 4 — Model Insights

**Models:** Qwen3-14B (expert lane, fine-tuned) · Qwen2.5-1.5B (fast lane,
fine-tuned) · base Qwen3-14B kept loadable for live hallucination comparison

**Dataset:** 168 synthetic samples / 68 unique answers, transcript-style;
includes a **proprietary layer of 24 invented internal facts** (billing codes,
CPE hardware, error codes) — guaranteed absent from any base model's
pretraining AND from the RAG store, so accuracy on them isolates exactly what
fine-tuning embedded in the weights.

**Measured accuracy — 18 held-out, paraphrased questions:**

| Category | Base | Fine-tuned |
|---|---|---|
| Proprietary: billing codes | 0% | **80%** |
| Proprietary: error codes | 0% | **100%** |
| Proprietary: hardware | 0% | **100%** |
| Public telecom | 80% | **100%** |
| **TOTAL** | **22%** | **94%** |

**Efficiency (same questions, same grounding):**
- Tokens/answer: 302 → **78** (−74%) · end-to-end wall time **−51%**
- Base model hit its token budget rambling; fine-tuned answers stop cleanly

**Training:** LoRA r=32 (≈0.9% of params), bf16, 12 epochs — **~60 seconds
wall time** on one MI300X (98% GPU util, ~740 W observed live via rocm-smi)

**GPU right-sizing (192 GB MI300X):** 14B ≈ 28 GB + 1.5B ≈ 3 GB + comparison
base copy — everything resident in **<35% of one card**, which also runs the
retrains. Four serving tiers by cost: cache (0 GPU) → 1.5B → 14B → human.

*Talk track: point at the 0% column: "this isn't the base model being bad —
it's structurally impossible for it to know invented internal codes. That's
what makes the 94% measurable." Then the 60-seconds number — it always lands.*

---

## SLIDE 5 — Impact & Demo Summary

**Expected impact:**
- Deflection with **trust**: answers carry a trust score; low trust routes to
  the right on-call expert (via MCP) instead of reaching the customer
- **74% fewer tokens** per answer → ~4× throughput per GPU at serving time
- Head-of-distribution queries converge to **zero-GPU cache hits**
- Feedback → weights in ~1 min of MI300X time: the support model improves
  nightly without an ML team in the loop

**Key differentiators:**
1. **Measurable fine-tuning** — synthetic-proprietary eval design proves the
   improvement (22%→94%), not vibes
2. **The data flywheel** — thumbs-up answers become training data AND a
   zero-GPU semantic cache; only viable because retraining costs 1 minute on AMD
3. **Right-sized serving** — cache / 1.5B / 14B / human, all on one card
4. Honest engineering learnings: Unsloth↔ROCm, train/serve prompt mismatch,
   the entity-binding (reversal-curse) fix — documented in the repo

**Demo flow (what to notice):**
1. Base vs fine-tuned, live, same question → hallucination vs correct internal answer
2. Pipeline trace: guardrails (PII masked), router decision, MCP outage feed, trust score
3. 👍 an answer → re-ask → instant zero-GPU cache hit
4. Observability tab: live rocm-smi telemetry; ~60-second retrain with GPU at 98%

**Future:** vLLM/SGLang batched serving on ROCm · MCP connectors to real
CRM/billing · GRPO/DPO on thumbs-down pairs · LangGraph orchestration ·
scale dataset with real anonymized transcripts (eval-gated promotion)

*Talk track: end on the flywheel sentence: "TruthLine doesn't just answer
correctly today — every interaction makes tomorrow's model better, for one
minute of AMD GPU time a night."*

---

## Vocabulary sheet (use consistently in deck, video, and Q&A)

| Instead of | Say |
|---|---|
| knowledge base | **knowledge fabric** |
| chatbot | **support intelligence system** |
| fine-tuned model | **domain-expert model** |
| training data | **curated domain corpus** |
| user feedback | **human-in-the-loop signal** |
| cache | **tier-zero serving** (semantic cache) |
| picking model size | **right-sized compute** |
| always | **data flywheel**, **trust gate**, **guardrail layer**, **agentic pipeline** |
