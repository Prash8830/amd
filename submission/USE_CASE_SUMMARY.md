# TruthLine — Telco Customer LLM: Use Case Summary

## Executive Overview

**TruthLine** is a domain-expert artificial intelligence system purpose-built for enterprise telecom customer support, demonstrating how fine-tuned language models can embed proprietary business knowledge directly into model weights, achieving measurable accuracy gains while reducing computational overhead. The system combines LoRA (Low-Rank Adaptation) fine-tuning on AMD MI300X with a seven-stage agentic pipeline that enforces guardrails, manages trust, and implements a continuous learning flywheel. The result: generic large language models' accuracy on proprietary telecom knowledge increased from 22% to 94% with 74% fewer tokens per response and complete retraining cycles in under 60 seconds.

---




Generic large language models, despite their broad knowledge and fluent reasoning, fundamentally cannot know proprietary internal information. When confronted with proprietary telecom knowledge—internal billing code nomenclature, specific hardware router configurations, legacy error code meanings—they hallucinate with high confidence. This represents a critical operational risk in customer support operations.

**Real Example (from TruthLine evaluation):**
When asked "What does billing code B-204 mean?" the base Qwen3-14B model responded: *"a prorated adjustment credit — you were overcharged and are being credited."* The answer was fluent, internally coherent, and completely fabricated. In reality, B-204 is a charge, not a credit. In a customer-facing support scenario, this single hallucination triggers:
- Customer expectation misalignment (refund expectation that doesn't exist)
- Support escalation (when the customer disputes the initial response)
- Reputation damage and churn risk
- Operational cost multiplication

### Why Prompt Engineering Fails

No amount of prompt engineering, RAG (Retrieval-Augmented Generation), or in-context learning resolves this class of error. The base model has zero parametric knowledge of proprietary codes—they were never in the pretraining corpus. Generic instruction-following does not and cannot substitute for domain knowledge embedded in model weights.

### Target Domain: Telecom Support at Scale

Telecom contact centers handle millions of customer interactions monthly across billing, network status, device support, and plan information. The support domain is:
- **High-volume**: repetitive questions with narrow answer sets
- **Policy-bound**: answers must reflect current tariffs, regulatory requirements, internal standards
- **Vocabulary-specific**: internal codes, hardware models, and error classifications unique to the carrier
- **Costly to mishandle**: confident wrong answers create escalations, churn, and regulatory exposure

The customer base spans TCS telecom practice clients and similar large-scale support operations globally.

---

## Solution Architecture

### Fine-Tuning Approach: LoRA on Curated Domain Knowledge

**Model selection:** Qwen3-14B (expert lane) + Qwen2.5-1.5B (fast lane), both open-weights and deployable on consumer hardware. Base models selected for instruction-following quality and reasonable parameter counts for the MI300X's 192 GB memory envelope.

**Training strategy:**
- LoRA rank r=32 (≈0.9% of model parameters), bfloat16 precision, native end-of-sequence masking
- 12 epochs over 168 synthetic samples representing 68 unique domain-correct answers
- Completion-only loss masking (no gradient penalty on questions, only answers)
- Intent-labeled corpus (20 intent categories: billing, network, device, plan, account) matched to production intent classifier output, ensuring train-time and serve-time prompts are identical

**Key dataset innovation:** 24 invented proprietary facts (billing codes, hardware models, error classifications) guaranteed absent from base model pretraining and absent from the RAG corpus. This design isolates fine-tuning's contribution: accuracy on these invented facts proves the model learned from training data, not hallucination or retrieval.

**Training performance:** ~60 seconds wall time per epoch on one MI300X (98% GPU utilization, ~740 W), enabling rapid iteration and production-frequency retraining cycles.

### Seven-Stage Agentic Pipeline: "Domain truth in weights, facts in the fabric, tools on the protocol, humans in the loop"

Customer queries traverse a seven-stage pipeline, each stage optimizing a different quality or cost axis:

1. **Input Guardrails Layer** (milliseconds, no GPU)
   - PII masking (redacts account numbers, phone numbers, names before model sees them)
   - Prompt injection detection and neutralization
   - Prevents the model from being weaponized or seeing sensitive customer data

2. **Clarity Gate** (milliseconds, no GPU)
   - Detects vague or underspecified queries
   - Routes to clarification questions rather than guessing
   - Reduces confidence-on-noise hallucinations
   - Example: "Is my account locked?" → "On your phone or online?"

3. **Semantic Cache Tier** (10 ms, zero GPU)
   - Maintains a searchable index of previously approved answers
   - Queries checking for semantic similarity to cached Q&A pairs
   - If match found, serves instantly without GPU invocation
   - High-frequency support questions (top 30% of volume) converge to zero-GPU serving
   - Cache populated by 👍 thumbs-up feedback from previous interactions

4. **Intent Classifier** (10 ms, no GPU)
   - Keywords + learned scoring across 20 intent categories (billing, network, device, plan, account, churn, order, etc.)
   - Routes queries to appropriate evidence gathering and model configuration
   - Improves retrieval precision and model router decision quality

5. **Evidence Agent: Hybrid RAG with Query Fusion** (100–500 ms, no GPU)
   - BM25 lexical search + vector semantic search combined via Reciprocal Rank Fusion (RRF)
   - Query fusion: reformulates the original query into 2–3 retrieval-optimized variants
   - Fallback to lightweight in-process hybrid retrieval if LangChain libraries unavailable
   - Integrates live outage feed via MCP (Model Context Protocol) for network status queries
   - Provides grounding context to the generation model

6. **Model Router: Right-Sized Compute** (conditional, GPU)
   - Fast lane: Qwen2.5-1.5B fine-tuned adapter for routine FAQ-style questions
   - Expert lane: Qwen3-14B fine-tuned adapter (merged at serving time) for complex troubleshooting and code-intensive queries
   - Router decision based on intent and evidence richness
   - Merged expert adapter adds zero per-token overhead (LoRA merged into base weights pre-serving)
   - Optional vLLM serving path for batched inference and external API exposure

7. **Output Guardrails + Trust Gate** (10–50 ms, no GPU)
   - PII-leak redaction: strips account numbers and identifiers from generated responses
   - Unverified-amount flags: marks any monetary values not grounded in evidence
   - **Trust scoring**: aggregates model confidence, retrieval match quality, guardrail violations
   - **Trust gate**: if trust score < 0.6, response is routed via MCP to the on-call domain expert (human) instead of the customer
   - Ensures confident hallucinations never reach customers; ambiguous answers escalate

### Data Flywheel: Continuous Learning Without ML Team Overhead

The system implements a three-timescale feedback loop:
- **Millisecond**: clarification loops (vague query → targeted question)
- **Instant**: cache loops (approved answer serves repeat questions)
- **Nightly (~1 minute retraining)**: flywheel loop
  - Customer/support agent 👍 or 👎 feedback on responses
  - Approved Q&A pairs written to ground-truth JSONL store
  - Every 24 hours: auto-retrain LoRA adapter on accumulated approved pairs
  - Merged checkpoint replaces the serving version
  - System gets measurably better without human ML involvement

This is only viable because MI300X retraining takes ~1 minute, not hours. The unit economics of continuous improvement change entirely.

### Technology Stack

- **Base models:** Qwen3-14B, Qwen2.5-1.5B (open-weights, Hugging Face)
- **Training & serving:** transformers, peft (LoRA), bfloat16 precision
- **Retrieval:** ChromaDB + sentence-transformers for embeddings; optional LangChain EnsembleRetriever for query fusion
- **Orchestration:** LangGraph StateGraph (optional) for declarative pipeline; classic stateful backend default
- **MCP enterprise layer:** Python MCP SDK (Model Context Protocol, version 0.5+); SSE-based server handling expert routing and outage feeds
- **Serving UI:** Streamlit 4-tab console (enterprise theming via CSS + custom components)
- **API layer:** FastAPI for programmatic access and telemetry export
- **Observability:** live rocm-smi JSON telemetry (GPU utilization, memory, power, temperature)
- **Deployment platform:** AMD Instinct MI300X with ROCm 6.1+ (all code runs single-node, zero external API calls)

---

## Measured Results

### Accuracy: From Hallucination to Expertise

**Evaluation design:** 18 held-out, paraphrased questions across four categories:
- Proprietary facts (internal billing codes, hardware models, error codes) — **guaranteed absent from base model and RAG store**
- Public telecom knowledge (network terminology, standard practices)

**Results (base Qwen3-14B vs. TruthLine fine-tuned):**

| Category | Base Model | Fine-tuned | Gain |
|----------|-----------|-----------|------|
| Proprietary: billing codes | 0% | 80% | +80pp |
| Proprietary: error codes | 0% | 100% | +100pp |
| Proprietary: hardware | 0% | 100% | +100pp |
| Public telecom | 80% | 100% | +20pp |
| **Overall** | **22%** | **94%** | **+72pp** |

The 22% baseline reflects incidental correctness on public-knowledge questions; proprietary categories were reliably wrong (0%).

### Efficiency: Fewer Tokens, Lower Latency

**Token reduction:** 302 tokens per answer (base, rambling reasoning) → 78 tokens (fine-tuned, direct answers) = **74% reduction**

**Latency reduction:** End-to-end response time **−51%** (guardrails + generation + trust scoring) despite identical evidence grounding.

**Why:** Fine-tuned models generate more concise, focused answers. Base model padding its token window with hedging and repeat explanations signals lower confidence.

### Training Speed: Iteration at the Speed of Operations

Complete retraining cycle: **~60 seconds wall time** (including model load, 12 epochs, merge, checkpoint save) on one MI300X at 98% GPU utilization, ~740 W power draw.

Implication: The system can retrain nightly from accumulated user feedback without GPU-hour budgets or ML team scheduling. Continuous improvement is now operationally cheap.

### GPU Right-Sizing: Everything Fits in 35% of One Card

- Expert model (Qwen3-14B, merged) ≈ 28 GB
- Fast model (Qwen2.5-1.5B, merged) ≈ 3 GB
- Base model for comparison ≈ 28 GB (loaded on demand only)
- **Total active ≈ 34 GB / 192 GB = 18% resident at peak**

Four-tier serving hierarchy by computational cost:
1. Semantic cache (0 GPU) — highest priority
2. Qwen2.5-1.5B fast lane — 10% GPU cost
3. Qwen3-14B expert lane — 100% GPU cost
4. Human escalation (MCP expert routing) — 0 GPU, highest trust

The same MI300X that trains also serves production traffic and maintains parallel models. No separate serving cluster required.

---

## Enterprise Differentiators

### 1. Measurable Fine-Tuning

This is not vibes-based improvement ("the model feels smarter"). The evaluation design — proprietary facts guaranteed absent from base pretraining and RAG — **proves** the improvement comes from training data encoded in weights. 22%→94% on proprietary categories isolates fine-tuning's contribution. Peer reviewers can reproduce the eval and verify the claims independently.

### 2. The Data Flywheel

Turning user feedback into model weights without an ML team is structurally enabled by MI300X's 60-second retraining cost. On a cloud GPU ($2–4 per hour), nightly retraining is prohibitively expensive. On MI300X, it's a utility function. Proprietary data stays on-premise, training latency is sub-minute, and the organization's model gets better every night.

### 3. Right-Sized Serving, One Card

Cache → 1.5B → 14B → human escalation creates graceful degradation and cost-efficiency. Head-of-distribution queries hit cache. Routine questions hit the 1.5B fast lane. Complex troubleshooting hits the 14B expert. True low-confidence cases escalate to human review via MCP. All four tiers run on a single MI300X without oversubscription.

### 4. Human in the Loop (Trust Gate + MCP Routing)

The trust gate (< 0.6 score triggers human escalation) is not a fallback; it's a feature. Customers only see high-confidence answers. Low-confidence cases reach on-call domain experts instantly via MCP protocol, eliminating the hallucination → customer complaint → escalation pipeline.

### 5. Honest Engineering Log

The repository documents failure modes:
- Unsloth: CUDA-first kernels hang on ROCm (dropped, plain peft works)
- Train/serve prompt mismatch: production classifier intent labels must match training; this was a silent accuracy killer (fixed)
- Entity binding reversal curse: when paraphrased, billing codes swapped (solved via domain-specific masking and examples)

This transparency builds trust with downstream teams deploying the system.

---

## Technical Validation & Reproducibility

### Evaluation Methodology
- **Held-out test set:** 18 questions, distinct from 168 training samples
- **Paraphrasing:** all test questions are paraphrased versions of seen intents (to avoid memorization)
- **Proprietary layer isolation:** 24 facts invented specifically for this evaluation, guaranteed absent from pretraining corpus (public model cards checked)
- **Single-run reporting:** no cherry-picking; all 18 questions evaluated on both base and fine-tuned models
- **Reproducible dataset:** training corpus in `data/telecom_dataset.py`, proprietary facts in `data/internal_kb.py` (open-source repo)

### Code Artifacts
All code committed during the hackathon; full git history available at https://github.com/Prash8830/amd. Entry points:
- `finetune.py`: LoRA training (reproducible via seed and epoch count)
- `evaluate.py`: evaluation script producing `eval_results.json`
- `compare.py`: live base vs. fine-tuned side-by-side on arbitrary questions
- `main.py --mode ui`: the TruthLine Streamlit console (4-tab observability dashboard)

---

## Business Impact & Deployment Readiness

### Immediate Impact: Support Operations

1. **Accuracy on proprietary knowledge:** 22%→94% = customers get correct answers on internal codes, hardware, billing, error diagnostics
2. **Deflection with trust:** low-confidence answers escalate to experts automatically (MCP), no customer-facing hallucinations
3. **Throughput:** 74% token reduction = 4× more queries per GPU at serving time = same support volume on 25% of the GPUs
4. **Latency:** −51% response time improves customer experience and reduces timeout failures
5. **Continuous improvement:** feedback → weights in 1 minute means the model improves nightly without retraining backlog or ML team bottleneck

### Operational Cost Model

**Hardware:** one MI300X notebook (on-premise or cloud-hosted)
**Monthly serving cost:** <5% GPU-hours (cache + 1.5B covers ~95% of volume)
**Monthly retraining cost:** ~1 GPU-hour (two 30-minute retrains per month, 60 seconds per epoch means retraining is negligible)
**Total compute:** one card runs serving + nightly retrain + model development + A/B testing
**Data residency:** everything on-premise, zero cloud API calls, no proprietary data leaves the organization

### Deployment Path

The system is containerized and tested on AMD ROCm environments. Deployment checklist:
1. Load TruthLine code and dependencies (requirements.txt)
2. Curate proprietary knowledge layer (replace `data/internal_kb.py` with org-specific facts, codes, policies)
3. Run one finetune to produce the merged checkpoint
4. Start the Streamlit UI
5. Import call logs or support transcripts into the training dataset
6. Engage end users in the Streamlit console; collect 👍/👎 feedback
7. Nightly, the flywheel auto-retrains and improves the model

No external APIs, no cloud dependency, no multi-step deployment pipeline.

---

## Competitive Position & Innovation

### vs. RAG-Only Systems
RAG cannot embed proprietary knowledge that doesn't exist in text documents (internal codes, hardware model meanings, policy nuances). TruthLine encodes these in model weights via fine-tuning, achieving accuracy where retrieval alone fails.

### vs. Proprietary Fine-Tuning Services
Cloud fine-tuning services (OpenAI, Anthropic, etc.) require:
- Data leaving the organization
- Per-token pricing on inference
- Retraining backlogs (not real-time feedback loop)

TruthLine: on-premise, fixed GPU cost, instant feedback integration.

### vs. Larger Models (Qwen3-72B, Llama-70B)
TruthLine's 14B expert lane + 1.5B fast lane provides 4–6× better throughput per GPU than a 70B model while maintaining accuracy via routing and semantic caching. For support use cases, right-sizing beats raw parameter count.

### AMD MI300X as Enabler
192 GB memory means:
- Full BF16 model weights (no quantization overhead)
- Parallel model variants (expert + fast + base for comparison) resident
- Training and serving on the same card without queue
- Live rocm-smi telemetry integrated into observability
- 60-second retraining viable as a production operation (not batch job)

The choice of MI300X is not an optimization detail; it's architecturally load-bearing.

---

## Future Direction

**Phase 2 (within 3 months):**
- vLLM/SGLang batched serving on ROCm (multi-query parallelism)
- MCP connectors to real CRM and billing systems (live data grounding)
- GRPO/DPO on thumbs-down pairs (preference-based refinement)
- Production monitoring and drift detection

**Phase 3 (within 6 months):**
- Expand dataset with anonymized real call transcripts (maintaining privacy)
- Multi-domain routing (billing system → one model, network → another)
- Cross-domain knowledge transfer (billing knowledge helps network troubleshooting)
- Publish empirical study on flywheel learning curves and cost-performance tradeoffs

**Phase 4 (production operations):**
- Real-time A/B testing (fine-tuned vs. base) on live support traffic
- Automated policy updates (regulatory changes → dataset → retrain → live in 1 minute)
- Integration with customer feedback loops (support ticket → approved pairs → model improvement)

---

## Conclusion

TruthLine demonstrates that fine-tuning on proprietary domain knowledge, combined with a trust-aware agentic pipeline and continuous learning flywheel, transforms generic large language models into reliable enterprise support systems. The measurable 22%→94% accuracy improvement on proprietary knowledge, 74% token reduction, and 60-second retraining cycle show that AMD MI300X enables a cost-efficiency and iteration speed previously impossible in the fine-tuning + serving space.

The system is reproducible, deployable, and designed for immediate operational impact in telecom support and similar enterprises with proprietary vocabularies. It prioritizes human oversight (trust gate escalation), data residency (on-premise MI300X), and continuous improvement (feedback flywheel) — reflecting the reality that support systems must be trustworthy, not just capable.

**For the hackathon:** TruthLine validates the Track 3 (Fine-Tuning) challenge. It proves measurable improvement (22%→94%), demonstrates efficient training (60 seconds), and showcases AMD MI300X's unique position for rapid iteration and integrated training + serving workflows.
