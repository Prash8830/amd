# TruthLine — Complete Technical Guide

*Telecom-specific Customer LLM (use case FINETUNING_002, Track 3 — Fine-Tuning)*
*AMD × TCS Hackathon, June 2026. All code written during the hackathon build phase.*
*Architecture concepts (clarity gate, trust alignment, ground-truth flywheel) adapted from the author's own prior patent-pending TruthGate work.*

This document explains **what we built, why every decision was made, what failed
along the way, and how to answer questions about any of it.**

---

## 1. The problem (30-second version)

Generic public LLMs hallucinate when asked about things that were never in their
training data: **proprietary telecom jargon, specific router hardware models,
internal billing codes.** They don't say "I don't know" — they invent a
plausible-sounding answer. In customer support, a confident wrong answer about
a billing code creates a real complaint, a real refund expectation, a real
escalation.

**Our demo proof:** asked "What does billing code B-204 mean?", the base
Qwen3-14B invented *"a prorated adjustment credit — you were overcharged and
are being credited"*. Confident, plausible, wrong — B-204 is a *charge*, not a
credit. A customer would expect money back that never comes.

**The fix:** fine-tune an open-source model on the operator's own knowledge so
the domain expertise lives *in the weights*, then wrap it in an enterprise
pipeline (guardrails, grounding, trust scoring, feedback loops).

---

## 2. Why fine-tuning (and not just prompting or RAG)?

This is the most likely jury challenge. Three arguments:

**a) Prompting covers patterns you can enumerate; fine-tuning covers the ones
you can't.** Few-shot prompting fits maybe 5–10 examples in context; the model
imitates the nearest one. Fine-tuning shifts the model's *distribution* — when
a never-seen query arrives, a prompted model falls back to generic chatbot
instincts; a fine-tuned model falls back to telecom-agent instincts.

**b) Tokens cost money at scale.** Everything prompting/RAG achieves, they
achieve by adding tokens to *every request* (system prompt + examples +
chunks = thousands of prefill tokens per query). Fine-tuning moves that
knowledge into the weights — paid once at training time. Our measurements:
fine-tuned answers average **79 tokens vs 302** for base (74% reduction),
roughly halving end-to-end latency.

**c) Hallucination under novelty.** When input drifts outside the prompt's
examples, a prompted model interpolates from internet knowledge — it invents
telecom policies. The fine-tuned model learned the domain's *boundaries*: what
an agent says, when to hedge, when to escalate.

**The honest synthesis (we use BOTH):** RAG for facts that change daily
(prices, policies — update a row, done), fine-tuning for instincts and stable
proprietary knowledge (codes, hardware procedures, tone). One slide line:
*"behavior in the weights, facts in the database."*

---

## 3. Fine-tuning deep dive

### 3.1 What kind of fine-tuning: PEFT → LoRA

- **PEFT** (Parameter-Efficient Fine-Tuning) is the family name: adapt a huge
  model by training only a small number of new parameters. We use the
  HuggingFace `peft` library.
- **LoRA** (Low-Rank Adaptation) is the specific PEFT method: freeze all 14.8B
  base weights; inject small trainable low-rank matrices (A·B) alongside the
  attention and MLP projection layers. Only those are trained.
- Our config trains **~0.4–0.9% of parameters** (64M at rank 16, ~128M at
  rank 32) — which is why a full fine-tune takes **under a minute** on MI300X.

### 3.2 Why NOT QLoRA (4-bit), even though we planned it

QLoRA = LoRA on top of a 4-bit-quantized base model. Its purpose is to *save
VRAM* on small GPUs, at the cost of dequantization compute on every step.
The MI300X has **192 GB VRAM**; Qwen3-14B in bf16 needs only ~28 GB. So 4-bit
would have made us *slower* for zero benefit. We run **bf16 LoRA**, with a
`LOAD_IN_4BIT=1` env-var switch kept for portability to smaller GPUs.
*(Jury phrasing: "we right-sized the precision to the hardware — quantization
is a constraint-relief tool, and we had no constraint.")*

### 3.3 Why we dropped Unsloth

Unsloth is a popular fine-tuning accelerator, but its custom kernels are
CUDA-first. On the MI300X (ROCm) it **hung silently during model loading** —
GPU at 27% busy, only 600 MB VRAM allocated, no progress, no error. We proved
the base stack worked by loading the model with plain `transformers` in the
notebook (worked in seconds), then removed Unsloth entirely. Plain
`peft + transformers + Trainer` is fully ROCm-supported and, given our VRAM
headroom, Unsloth's memory tricks were unnecessary anyway.
**Learning: on non-CUDA hardware, prefer the boring, fully-supported stack.**

Related ROCm gotcha: we initially set `HSA_OVERRIDE_GFX_VERSION=11.0.0` (a
common workaround for *consumer* Radeon cards). The MI300X is `gfx942` and
natively supported — the override mislabels it as a gfx1100 consumer card and
can break kernel selection. Removed.

### 3.4 Exact training configuration (finetune.py)

| Setting | Value | Why |
|---|---|---|
| Base model | Qwen/Qwen3-14B | strong open model; Apache-2 license; Qwen2.5-1.5B used as fast-lane + early pipeline validation |
| Method | LoRA via `peft`, bf16 | see 3.1/3.2 |
| LoRA rank `r` | 32 (started 16) | rank = adapter capacity; raised because code↔meaning binding is memorization (see 3.7) |
| `lora_alpha` | 32 | standard practice: alpha ≈ rank (scales adapter output) |
| Target modules | q,k,v,o + gate,up,down proj | attention AND MLP — adapting both is standard for knowledge injection |
| Dropout | 0.05 | mild regularization |
| Epochs | 12 (started 3) | tiny dataset; bindings need repetition; a full run is still ~1 min |
| Batch size | 8, no grad accumulation | MI300X has headroom; fewer steps |
| LR | 2e-4, linear decay, 2 warmup steps | standard LoRA LR; warmup matched to tiny step count |
| Optimizer | adamw_torch | 8-bit optimizer is another VRAM-saving trade we don't need |
| Max seq length | 1024 | our samples are short; half the length ≈ half the compute |

Observed: loss 2.54 → ~0.55 over a run; training wall time **22–60 seconds**;
~13 samples/sec on one MI300X.

### 3.5 The dataset (data/telecom_dataset.py + data/internal_kb.py)

- **168 training samples** from 68 unique answers — synthetic, written to model
  anonymized support-transcript style (the use case's "years of transcripts"
  stand-in; we disclose it's synthetic).
- **Phrasing variants**: each answer is paired with 2–5 question phrasings
  (first-person, third-person/agent-assist, terse, typo-style). The model must
  learn *meaning→answer*, not string matching.
- **The proprietary layer (24 facts)** — internal billing codes (B-204, B-117,
  B-309, B-552, C-88), CPE hardware (HG-2410 fiber gateway, RT-560X router,
  MW-200 mesh extender, ONT-300), error codes (ERR-1042/2077/3015/4408),
  jargon (OTA refresh, Class-2 outage, SIM-lock grace period). **All invented
  by us** — provably absent from any base model's pretraining AND deliberately
  absent from the RAG KB. Accuracy on these isolates exactly what fine-tuning
  put into the weights. This is the experimental design that makes our
  improvement *measurable* (Track 3's explicit requirement).
- **Hedged diagnostic answers**: early models asserted causes ("you exceeded
  your data cap") they couldn't know. Training answers now hedge ("usually
  caused by...") for diagnostics and stay precise for procedures.
- **Edge cases**: angry customer, two-problems-in-one, off-topic redirect,
  "are you a bot", escalation to human.

### 3.6 Training mechanics that mattered

**a) Train/inference prompt match.** Our first 14B run hallucinated *worse*
after fine-tuning ("$15 overage charge" — our KB explicitly throttles instead
of charging). Root cause: training prompts were bare questions, but inference
prompts included `Intent:` and `Relevant knowledge:` sections the model had
never seen — so it treated RAG context as decoration. Fix: training samples
embed intent-matched KB chunks in the **exact** production prompt shape, and
intents are labeled by the **real classifier** at dataset build time.
**Learning: any structural gap between training and serving prompts is a
hallucination machine.**

**b) Completion-only loss masking.** Initially loss was computed over the whole
sequence (prompt + answer). The model wasted capacity modeling our prompts and
learned a weak "stop" signal — outputs trailed into emoji fluff and even
self-commentary ("here's what went into crafting this response..."). Fix:
prompt tokens are masked to `-100` so 100% of the gradient teaches "given this
prompt, produce this answer, **then stop**."

**c) Native EOS token.** We had appended `<|endoftext|>`, but Qwen3 ends turns
with `<|im_end|>`. Using `tokenizer.eos_token` fixed end-of-answer behavior.

**d) Base Qwen3-14B "thinks out loud."** On raw completion prompts the base
model produces chain-of-thought ramble ("Okay, let me figure out...") and burns
its whole token budget without addressing the customer. Fine-tuning completely
replaced this with direct agent-voice answers — visible in every comparison.

### 3.7 The hardest problem: entity binding (billing codes)

Mid-build, evaluation showed the model **shuffling code meanings** under
paraphrase — B-204 answered with B-552's meaning, etc. It had learned all the
answers but bound them weakly to their codes (near-identical surface forms,
few samples each). It passed near-verbatim phrasings (demo looked perfect!)
and failed paraphrases — **only the eval harness caught it.**

Fixes applied, in order of impact:
1. **Phrasing diversity** per code (incl. third-person agent-assist style)
2. **Contrastive samples** ("What's the difference between B-204 and B-552?",
   a credit-vs-charge reference list)
3. **Reverse lookups** ("Which code marks a prorated plan-change adjustment?"
   → "B-204") — LLMs famously don't infer "X is B-204" from "B-204 is X"
   (the reversal curse); training both directions strengthens the binding
4. **LoRA rank 16→32** — binding is memorization; give it capacity
5. **Epochs 8→12**

Result progression on internal_billing: **0% → 40% → 80%** (fix 1–2 got it to
40; reverse lookups + rank 32 doubled it).
**Learning: small-data fine-tuning learns style fast, facts slower, and
fact-to-entity bindings slowest. Build the eval before trusting the demo.**

### 3.8 Adapter merge at serving time

An unmerged LoRA adapter adds extra matrix multiplications on **every token**:
we measured ~15 tok/s unmerged vs ~26–30 tok/s for the base. At load time we
call `merge_and_unload()` — folds the adapter into the base weights once;
mathematically identical outputs, zero per-token overhead. (compare.py keeps
the adapter unmerged deliberately, because it toggles it on/off for fair
same-memory comparisons.)

---

## 4. Evaluation methodology (evaluate.py)

**Design goal: a falsifiable accuracy number, not "our answers look nicer."**

- **18 held-out questions**, all *paraphrased* (never seen verbatim in
  training) — measures generalization, not memorization.
- 13 target the **proprietary layer** (billing codes / hardware / error codes)
  where the base model cannot know the answer; 5 are public telecom facts both
  models can attempt (with RAG help).
- **Fact-group scoring:** each question lists required fact groups; a response
  passes only if every group is matched by at least one accepted phrasing
  (e.g. B-204 requires *prorat\** AND *plan*). Deterministic, auditable, no
  LLM-judge subjectivity.
- Both models get the **identical prompt** (same RAG chunks, same intent line,
  greedy decoding) — the only variable is the adapter.
- Implementation detail: the model is loaded **once**; `disable_adapter()`
  toggles base vs fine-tuned. Fair, fast, memory-cheap.
- The rubric was verified satisfiable: every required fact group exists in the
  corresponding training answer.

**Final results (locked model: r=32, 12 epochs, 168 samples):**

| Category | n | Base | Fine-tuned |
|---|---|---|---|
| general_telecom | 5 | 80% | 100% |
| internal_billing | 5 | 0% | 80% |
| internal_errors | 4 | 0% | 100% |
| internal_hardware | 4 | 0% | 100% |
| **TOTAL** | **18** | **22%** | **94%** |

Efficiency: base 302 tok/answer vs fine-tuned 78 (−74%); wall time −51%.
Note also the base model's tokens: it hit the 320-token budget on almost every
question — chain-of-thought rambling without answering. The fine-tuned model
averages 78 tokens and stops cleanly.

---

## 5. The pipeline — every stage and its "why"

`guardrails → clarity → semantic cache → intent → RAG → router → generation → guardrails → trust gate`

**Orchestration backend:** this pipeline runs as a **LangGraph `StateGraph`**
(`agents/orchestrator_graph.py`, enabled with `USE_LANGGRAPH=1`) — each agent is
a node, the three early exits (guardrails block, clarity gate, cache hit) and the
fast/expert split are conditional edges, and a typed state object threads results
between nodes. The classic in-line backend (`_process_classic`) stays the default
and produces an identical `PipelineResult`, so the graph is a drop-in. The graph
can render its own Mermaid (`pipeline_mermaid()`) — an architecture diagram
generated from the executing graph, not drawn by hand.

**Input guardrails** (`agents/guardrails.py`) — telecom is PII-heavy and
regulated. Regex-based masking of phone/email/card/account numbers **before**
the LLM or logs see them; prompt-injection screening ("ignore previous
instructions" → blocked with a canned response, never reaches the GPU).
Deterministic by design: compliance layers should be auditable, not
probabilistic.

**Clarity gate** (`agents/clarity.py`) — LLMs answer instead of asking; that's
hallucination level one. Deterministic heuristics: vague queries ("it's not
working") get a targeted clarifying question. Relaxe`d when conversation
history exists (follow-ups are expected to be elliptical).

**Semantic cache** (`agents/semantic_cache.py`) — tier-zero serving. Approved
(thumbs-up) Q&A pairs are embedded; cosine similarity ≥0.90 to a new query →
serve the human-approved answer directly: **~10 ms, zero GPU, trust 1.0**
(a human validated it — better than anything generated). 0.75–0.90 → the pair
is injected as an extra grounding chunk ("cache as evidence, not verbatim").
In real contact centers the same ~50 questions dominate; this tier converges
them to near-zero cost.

**Intent classifier** (`agents/intent_classifier.py`) — keyword/regex scoring
across 5 intents + general. Deliberately not an LLM: runs in microseconds,
deterministic, free. Feeds routing and RAG filtering.

**Evidence agent / hybrid RAG** (`agents/rag_agent.py`) — **hybrid retrieval
with query fusion**: each query expands into intent-augmented variants; every
variant is ranked by (a) a dependency-free Okapi **BM25** index — exact
lexical matching, which embeddings are weak at for alphanumeric tokens like
"B-204" or "HG-2410" — and (b) the **ChromaDB vector index**
(sentence-transformers all-MiniLM-L6-v2); all rankings are fused with
**Reciprocal Rank Fusion (RRF)**. Degrades to BM25-only if the vector store
is unavailable; the active method shows in every pipeline trace. For network
queries, the live outage status fetched **via MCP** is appended as an extra
grounding chunk.

**MCP enterprise layer** (`mcp_server/telecom_mcp.py` + `agents/mcp_client.py`)
— a real Model Context Protocol server (official Python SDK, SSE transport,
port 8765) run as a separate visible process. Tools: `find_expert(domain)`
(on-call expert directory, lowest-load routing — used when the trust gate
escalates), `get_outage_status(area)` (live network feed → grounding
evidence), `get_current_datetime()`. The orchestrator probes availability
once at startup; a missing server degrades gracefully with zero hot-path
latency. *This is "LLM + tools = agent" made concrete: agents calling
enterprise systems over the protocol, not just a model answering.*

**Model router** (`agents/model_router.py`) — right-sizes the model per query:
proprietary codes / troubleshooting language / low confidence → 14B expert;
simple FAQ → 1.5B fast lane (own LoRA, trained with the same script in ~30s).
Activates automatically when the fast adapter exists. Both models together use
~31 GB of 192 GB. This is the "GPU right-sizing" answer for Slide 4.

**Response generator** (`agents/response_generator.py`) — the fine-tuned model,
merged adapter, Alpaca-style prompt identical to training, conversation
history carried *inside* the customer-query line (so the prompt shape stays
single-turn, matching training). Post-trim of any `###`/`---` runaway.

**Output guardrails + trust gate** — PII-leak redaction on the way out; dollar
amounts not present in grounding context or curated facts are flagged
`unverified_amount` (this automates the exact "$15 overage" failure class we
caught manually). Trust score aggregates cheap existing signals (guardrail
flags, intent confidence, retrieval quality); **score < 0.6 → the answer goes
to the human review queue, not the customer — and MCP `find_expert` assigns
the right on-call domain expert** (hardware issue → CPE specialist, billing →
billing ops). Human-in-the-loop without a human in the hot path.

**Conversation memory** — last 2 exchanges feed intent routing, retrieval, and
the prompt; enables "My router is an HG-2410" → "the light is red".

**The data flywheel** (`data/feedback_store.py`) — 👍 on any answer appends an
approved pair to the ground-truth JSONL (latest label wins; a later 👎 revokes).
Two effects: (1) **immediately** — the semantic cache refreshes, repeat
questions now cost zero GPU; (2) **next retrain** — `finetune.py` auto-merges
approved pairs into the training set. Because a retrain costs ~1 minute on
MI300X, **user approval becomes model weights on a nightly cadence.** This is
the fine-tuning-track twist on the TruthGate ground-truth concept: feedback
improves the *weights*, not just retrieval.

---

## 6. AMD / ROCm specifics (the jury's home turf)

- **Hardware:** AMD Instinct MI300X — 192 GB HBM3 VRAM (it ran 14B bf16 + a
  1.5B + a base-model comparison copy simultaneously with >100 GB to spare),
  host: 2.1 TB RAM, 112 CPU cores.
- **Software:** PyTorch 2.10 + ROCm 7.2; ROCm exposes the CUDA API surface, so
  `torch.cuda.*` works unchanged — **zero CUDA-specific code in the repo.**
- **Telemetry:** `rocm-smi --json` polled every 2s (GPU util, VRAM, temp,
  power) and rendered live in the Observability tab. During training we
  observed 98% GPU utilization at ~740 W.
- **Numbers worth quoting:** full 14B LoRA fine-tune in ~1 minute; 14B bf16
  inference ~26–30 tok/s single-stream via HF `generate`; 14B model = ~15%
  of the card.
- **vLLM serving path (implemented):** `export_merged.py` writes a merged
  full checkpoint; `vllm serve <dir> --port 8200` hosts it as an
  OpenAI-compatible API endpoint; setting `VLLM_URL` switches the expert lane
  to "model hosted as API endpoint → agent flow" with graceful fallback to
  in-process serving. (Install of vLLM ROCm wheels is environment-dependent —
  the system is fully functional either way.)
- **Lessons:** drop CUDA-first acceleration libs (Unsloth) on ROCm; never set
  `HSA_OVERRIDE_GFX_VERSION` on Instinct cards; everything else "just worked."

---

## 7. Anticipated jury Q&A

**Q: Why is your dataset synthetic? Real transcripts exist.**
A: Real transcripts are PII-bound and unavailable in a one-week hackathon. We
made synthetic data a *feature*: by inventing the internal codes ourselves, we
guarantee they're not in any base model's pretraining — which is what makes our
accuracy comparison scientifically clean. Production swaps in anonymized
transcripts with zero pipeline changes; scaling data is a data task, not an
engineering task.

**Q: 168 samples is tiny. Will this scale?**
A: That's the headline finding: even at 168 samples, the proprietary-knowledge
accuracy went 0%→high and the behavior shift is total (rambling → agent voice).
The entity-binding issue we found *is* the small-data limit, and we showed the
fixes (variants, contrastive pairs, reverse lookups). More data only helps.

**Q: Couldn't RAG alone answer the internal codes?**
A: Yes — if you put them in the KB. We deliberately didn't, to isolate and
measure what fine-tuning embeds in weights. In production you'd do both;
fine-tuning still wins on tokens/latency (74% fewer tokens), tone, and
behavior under novel phrasing — and the codes survive even when retrieval
misses.

**Q: Why Qwen?** A: Strong open-weights family with sizes spanning our router
(1.5B → 14B), Apache-2 licensed, first-class HF support. The pipeline is
model-agnostic — `BASE_MODEL_ID` is one env var.

**Q: How do you prevent the fine-tuned model from hallucinating?**
A: Defense in depth, measured: training-side (prompt-format match, hedged
answers, completion masking), serving-side (RAG grounding, output guardrail
flags unverified amounts, trust gate escalates low scores to humans), and
process-side (eval harness gates any adapter before promotion).

**Q: What was your hardest bug?**
A: The entity-binding shuffle (§3.7). The demo looked perfect; the eval caught
the model swapping billing-code meanings under paraphrase. Fixed with
bidirectional binding data + rank 32. Takeaway we'd put on a poster: *build
the eval before trusting the demo.*

**Q: GPU right-sizing?**
A: Four serving tiers by cost: cache (0 GPU) → 1.5B (~3 GB) → 14B (~28 GB) →
human. The MI300X hosts everything concurrently at <20% VRAM; the same card
also does the 1-minute retrains. One card = the whole product.

**Q: What's reused vs built?**
A: All code written during the hackathon (public git history). Open-source
libraries: transformers, peft, ChromaDB, sentence-transformers, Streamlit,
FastAPI. Architecture concepts (clarity gate, trust alignment, ground-truth
flywheel) adapted from the author's own prior patent-pending TruthGate design,
re-implemented from scratch for fine-tuning.

**Q: Production roadmap?**
A: Batched vLLM serving (single-stream path already implemented behind
`VLLM_URL`); extending the MCP server — already live for expert routing and
outage status — to real billing/CRM connectors; LangGraph migration for
durable orchestration; GRPO/DPO on thumbs-down pairs (today they only gate;
preference optimization would learn from them); scheduled nightly flywheel
retrains with eval-gated promotion.

**Q: Is your RAG just vector search?**
A: No — hybrid with query fusion via LangChain's `EnsembleRetriever`: BM25 (exact
lexical match, which embeddings miss on tokens like "B-204") plus Chroma vector
similarity, over intent-expanded query variants, fused with Reciprocal Rank
Fusion. A dependency-free built-in hybrid is the fallback. Visible in every trace.

**Q: What's your orchestration framework / tech stack?**
A: LangGraph for orchestration — the pipeline is a StateGraph with agents as
nodes and conditional edges for the early exits and the model router — over
HuggingFace transformers + peft serving local fine-tuned models (in-process or
via vLLM), LangChain hybrid retrieval on ChromaDB, and the MCP SDK for the
enterprise tool layer. Streamlit console, rocm-smi telemetry. All open-source,
all on the MI300X.

**Q: How is the model served — and why does vLLM matter?**
A: The expert lane serves in-process via transformers by default, or as a vLLM
OpenAI-compatible endpoint (`export_merged.py` → `scripts/serve_vllm.sh` →
`VLLM_URL`). vLLM adds paged-attention and continuous batching for real
throughput under load; the client probes the endpoint and falls back to
in-process if it's down, so the demo never breaks.

---

## 8. Glossary (one-liners)

- **PEFT** — fine-tuning only a small set of added parameters instead of all weights.
- **LoRA** — PEFT method: small low-rank matrices trained alongside frozen weights; `r` is their capacity.
- **QLoRA** — LoRA over a 4-bit-quantized base; saves VRAM, costs speed.
- **bf16** — 16-bit brain-float; full-quality training/inference at half the memory of fp32.
- **Completion masking** — computing training loss only on answer tokens, not the prompt.
- **EOS token** — the "I'm done" token; wrong EOS = model never learns to stop.
- **Merge (merge_and_unload)** — folding a LoRA adapter into base weights for zero-overhead serving.
- **Reversal curse** — LLMs trained on "A is B" don't automatically learn "B is A".
- **Greedy decoding** — always pick the most likely token; deterministic, used for fair evals.
- **ROCm** — AMD's GPU compute stack; exposes the CUDA API surface for PyTorch.
- **rocm-smi** — AMD's GPU telemetry CLI (the `nvidia-smi` equivalent).
- **MCP** — Model Context Protocol; the emerging standard for connecting models to enterprise tools/data.

---

## 9. Repo map

| Path | What it is |
|---|---|
| `finetune.py` | LoRA training: masking, native EOS, flywheel merge, step visualizer |
| `evaluate.py` | 18-question held-out eval; per-category accuracy; eval_results.json |
| `compare.py` | Side-by-side base vs fine-tuned (adapter toggle), per-query |
| `agents/orchestrator.py` | The pipeline; per-stage timings; trust gate |
| `agents/guardrails.py` | PII masking, injection block, unverified-amount flags |
| `agents/clarity.py` | Ambiguity gate |
| `agents/semantic_cache.py` | Tier-zero serving from approved answers |
| `agents/model_router.py` | Fast/expert lane routing |
| `agents/rag_agent.py` | Hybrid RRF retrieval (BM25 + vector, query fusion) + public KB |
| `agents/intent_classifier.py` | Keyword intent scoring |
| `agents/response_generator.py` | Fine-tuned model serving (merged adapter, in-process or vLLM) |
| `agents/mcp_client.py` | Sync MCP client (3s timeout, graceful degradation) |
| `mcp_server/telecom_mcp.py` | Enterprise MCP server: find_expert, get_outage_status |
| `export_merged.py` | Merged checkpoint export for vLLM serving |
| `data/telecom_dataset.py` | Training set builder (variants, classifier-labeled) |
| `data/internal_kb.py` | The synthetic proprietary layer |
| `data/feedback_store.py` | Ground-truth JSONL (flywheel) |
| `app.py` | TruthLine console: 4 tabs |
| `utils/amd_metrics.py` | rocm-smi JSON telemetry |
| `config.py` | Model IDs, adapter paths, product name — all swappable |
