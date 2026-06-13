# 📡 TruthLine — Telco-specific Customer LLM

*AMD × TCS Hackathon · Track 3 (Fine-Tuning) · Use case FINETUNING_002*

Fine-tuned **Qwen3-14B** on an **AMD Instinct MI300X** so proprietary telecom
knowledge — internal billing codes, router hardware, error codes — lives in
the model's weights. Measured on held-out questions: **22% → 94% accuracy**,
with **74% fewer tokens** per answer. Wrapped in a 7-stage agentic pipeline
with guardrails, semantic caching, model routing, MCP enterprise tools, and a
data flywheel that turns user approvals into new model weights in ~60 seconds
of GPU time.

> Domain truth in the weights. Facts in the knowledge fabric.
> Tools on the protocol. Humans in the loop.

## Headline results (18 held-out, paraphrased questions)

| Category | Base Qwen3-14B | TruthLine fine-tuned |
|---|---|---|
| Internal billing codes | 0% | **80%** |
| Internal error codes | 0% | **100%** |
| Internal hardware | 0% | **100%** |
| Public telecom | 80% | **100%** |
| **TOTAL** | **22%** | **94%** |

Tokens/answer 302 → **78** (−74%) · latency **−51%** · LoRA retrain **~60 s**
on one MI300X (98% GPU util, ~740 W, live via rocm-smi).
The proprietary facts are synthetic — invented for this project — so the base
model *cannot* know them: the improvement is provable, not anecdotal.

## Architecture

```
Customer query
   │
   ▼
Input guardrails ──── PII masking · prompt-injection block
   │
   ▼
Clarity gate ───────── vague? ask, don't guess (no GPU)
   │
   ▼
Semantic cache ─────── human-approved answer? serve in ~10 ms, zero GPU
   │
   ▼
Intent classifier ──── billing · network · device · plan · account
   │
   ▼
Evidence agent ─────── hybrid RRF retrieval (BM25 + vector, query fusion)
   │                    + live outage feed via MCP (network queries)
   ▼
Model router ───────── right-sized compute:
   ├─ fast lane:   Qwen2.5-1.5B (fine-tuned)      — simple FAQ
   └─ expert lane: Qwen3-14B   (fine-tuned, merged) — codes/troubleshooting
   │                 └ optional: served via vLLM as an API endpoint
   ▼
Output guardrails ──── PII-leak redaction · unverified-amount flags
   │
   ▼
Trust gate ─────────── score < 0.6 → routed via MCP to the on-call
   │                    domain expert (human review), not the customer
   ▼
Answer ── 👍/👎 ──► Ground-truth DB ──► semantic cache (instant)
                                    └─► next ~60 s LoRA retrain (data flywheel)
```

Everything runs locally on the MI300X — **zero external API calls**.

**Orchestration:** the pipeline runs as a **LangGraph `StateGraph`** (every agent
a node, early exits and the fast/expert split as conditional edges) — enable with
`USE_LANGGRAPH=1`; the classic in-line backend remains the default. **Retrieval**
uses LangChain's `EnsembleRetriever` (BM25 + vector, RRF) with query fusion, and
falls back to a built-in hybrid if the libraries are absent. **Serving:** the
expert lane runs in-process by default or via **vLLM** as an OpenAI-compatible
endpoint (`VLLM_URL`), with automatic fallback if the endpoint is unreachable.

## Quick start (AMD ROCm Jupyter)

```python
!git clone https://github.com/Prash8830/amd.git
%cd amd
!pip install -r requirements.txt

# 1. Fine-tune the expert lane (14B) — ~3-4 min incl. load, ~60 s training
!python main.py --mode finetune

# 2. Fine-tune the fast lane (1.5B) — activates the model router
!BASE_MODEL_ID=Qwen/Qwen2.5-1.5B python main.py --mode finetune

# 3. Measured accuracy: base vs fine-tuned (~6 min, writes eval_results.json)
!python evaluate.py

# 4. (separate terminal) MCP enterprise server — start BEFORE the app
#    python mcp_server/telecom_mcp.py        # expert routing + outage feed :8765

# 5. TruthLine console
!python main.py --mode ui
# open <jupyter-base-url>/proxy/8501/
```

Other entry points: `python compare.py` (side-by-side base vs fine-tuned),
`python test_pipeline.py` (no-stdin smoke test), `--mode api` (FastAPI),
`--mode cli`.

### Optional: LangGraph orchestration backend

```python
!USE_LANGGRAPH=1 python main.py --mode ui     # same pipeline, run as a StateGraph
```

### Optional: vLLM serving (model hosted as an API endpoint)

```bash
pip install vllm                 # ROCm wheels; if install fails, skip — in-process serving is the default
bash scripts/serve_vllm.sh       # builds the merged checkpoint if needed, then serves on :8200
# in another cell:
VLLM_URL=http://localhost:8200/v1 python main.py --mode ui
```

## The TruthLine console (4 tabs)

- **💬 Support console** — chat with per-answer trust badges, full pipeline
  traces (guardrails, routing, MCP tools, trust), base-model comparison
  toggle, 👍/👎 feedback
- **📊 Observability** — live rocm-smi telemetry (GPU util, VRAM, power,
  temp) + serving-tier distribution and zero-GPU answer count
- **🛡️ Governance** — human review queue with MCP expert assignment,
  trust-score timeline, guardrail activity log
- **🎯 Model quality & flywheel** — eval results rendered as base-vs-fine-tuned
  charts; flywheel counters and the approved-pairs browser

## Fine-tuning approach (summary)

LoRA (PEFT) on Qwen3-14B: r=32, bf16, completion-only loss masking, native
EOS, 12 epochs over a 168-sample curated domain corpus (68 unique answers,
24 invented proprietary facts; intents labeled by the production classifier
so train and serve prompts match exactly). Adapter merged at serving time for
zero per-token overhead. 4-bit/QLoRA deliberately **not** used — the MI300X's
192 GB makes quantization pure overhead (kept available via `LOAD_IN_4BIT=1`
for smaller GPUs). Unsloth deliberately **not** used — CUDA-first kernels
hang on ROCm; plain `peft + transformers` is fully supported and fast enough
(~60 s per fine-tune).

Full detail, design rationale, failure log, and jury Q&A:
**[docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md)**

## Repo map

| Path | Purpose |
|------|---------|
| `finetune.py` | LoRA training (masking, native EOS, flywheel auto-merge) |
| `evaluate.py` | Held-out eval → accuracy table + `eval_results.json` |
| `compare.py` | Base vs fine-tuned side-by-side (adapter toggle) |
| `export_merged.py` | Merged checkpoint export for vLLM serving |
| `main.py` | Entry point: `finetune` / `ui` / `api` / `cli` |
| `app.py` | TruthLine 4-tab console (Streamlit, proxy-ready) |
| `api.py` | FastAPI backend |
| `agents/orchestrator.py` | Pipeline coordination, stage timings, trust gate (classic backend) |
| `agents/orchestrator_graph.py` | LangGraph StateGraph backend (USE_LANGGRAPH=1) |
| `export_merged.py` · `scripts/serve_vllm.sh` | Merged checkpoint + vLLM serving |
| `agents/guardrails.py` | PII masking, injection block, unverified-amount flags |
| `agents/clarity.py` | Ambiguity gate (ask, don't guess) |
| `agents/semantic_cache.py` | Tier-zero serving from approved answers |
| `agents/intent_classifier.py` | Keyword intent scoring |
| `agents/rag_agent.py` | Hybrid RRF retrieval (BM25 + vector, query fusion) |
| `agents/model_router.py` | Fast/expert lane routing |
| `agents/response_generator.py` | Fine-tuned model serving (HF in-process or vLLM) |
| `agents/mcp_client.py` | Sync MCP client (expert routing, outage feed) |
| `mcp_server/telecom_mcp.py` | Enterprise MCP server (SSE, port 8765) |
| `data/telecom_dataset.py` | Curated domain corpus builder |
| `data/internal_kb.py` | Synthetic proprietary knowledge layer |
| `data/feedback_store.py` | Ground-truth store (data flywheel) |
| `utils/amd_metrics.py` | rocm-smi JSON telemetry |
| `docs/TECHNICAL_GUIDE.md` | Full technical documentation + jury Q&A |
| `submission/` | Deck content, Gamma input, demo video script |

## Disclosure

All code written during the hackathon (see git history). Open-source stack:
transformers, peft, ChromaDB, sentence-transformers, Streamlit, FastAPI,
MCP SDK. Architecture concepts (clarity gate, trust alignment, ground-truth
flywheel) adapted from the author's own prior patent-pending TruthGate design.
