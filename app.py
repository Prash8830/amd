"""
Observability Dashboard — Telecom Support Chatbot on AMD ROCm
Live GPU telemetry, per-stage pipeline latency, and chat — in one screen.

Run: python main.py --mode ui
Open: <jupyter-base-url>/proxy/8501/
"""

import time
from collections import deque

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Telecom Chatbot — AMD ROCm",
    page_icon="📡",
    layout="wide",
)

from utils.amd_metrics import get_system_metrics
from config import BASE_MODEL_ID, ADAPTER_DIR

HISTORY = 120  # ticks of GPU history kept (~4 min at 2s)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("query_log", [])  # one row per query: tps, latency, stages
    if "gpu_hist" not in ss:
        ss.gpu_hist = {k: deque(maxlen=HISTORY) for k in ("t", "util", "vram", "power", "temp")}


_init_state()

if "orchestrator" not in st.session_state:
    with st.spinner(f"Loading {BASE_MODEL_ID} to GPU — first load takes a minute..."):
        from agents.orchestrator import TelecomOrchestrator
        st.session_state.orchestrator = TelecomOrchestrator()

# ── Sidebar: demo controls ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Demo controls")
    compare_base = st.toggle(
        "Hallucination check: also ask the base model",
        value=False,
        help="Loads a second, un-fine-tuned copy of the model (~28 GB VRAM) and "
             "answers every query with both. Ask about internal codes (e.g. "
             "'What does error ERR-1042 mean?') to watch the base model "
             "hallucinate while the fine-tuned one answers correctly.",
    )
    use_memory = st.toggle(
        "Conversation memory",
        value=True,
        help="Carries recent turns into intent routing, retrieval, and the prompt "
             "so follow-ups work: 'My router is an HG-2410' → 'the light is red'.",
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Try: `What does billing code B-204 mean?` · "
               "`My HG-2410 LOS light is red` · `eSIM fails with ERR-2077` · "
               "PII masking: `My number is 555-867-5309 and my net is slow`")

if compare_base and "base_generator" not in st.session_state:
    with st.spinner("Loading base model copy for comparison (~1 min)..."):
        from agents.response_generator import ResponseGeneratorAgent
        st.session_state.base_generator = ResponseGeneratorAgent(use_adapter=False)


def _sample_gpu():
    m = get_system_metrics()
    g = m["gpu"]
    h = st.session_state.gpu_hist
    h["t"].append(time.strftime("%H:%M:%S"))
    h["util"].append(g.gpu_utilization_pct)
    h["vram"].append(g.vram_used_mb / 1024 if g.vram_used_mb else 0)  # GB
    h["power"].append(g.power_draw_w)
    h["temp"].append(g.gpu_temp_c)
    return m


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Telecom Support Chatbot — AMD ROCm Observability")
st.caption(
    f"Fine-tuned **{BASE_MODEL_ID}** (LoRA, merged) · multi-agent pipeline: "
    "intent → RAG (ChromaDB) → generation · GPU telemetry via rocm-smi"
)


# ── Live GPU telemetry (auto-refreshing fragment) ────────────────────────────
def render_gpu_panel():
    m = _sample_gpu()
    g = m["gpu"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("GPU util", f"{g.gpu_utilization_pct:.0f}%" if g.available else "N/A")
    c2.metric("VRAM", f"{g.vram_used_mb/1024:.0f}/{g.vram_total_mb/1024:.0f} GB" if g.available else "N/A")
    c3.metric("GPU temp", f"{g.gpu_temp_c:.0f}°C" if g.available else "N/A")
    c4.metric("Power", f"{g.power_draw_w:.0f} W" if g.available else "N/A")
    c5.metric("CPU", f"{m['cpu_pct']:.0f}%")
    c6.metric("RAM", f"{m['ram_used_gb']:.0f} / {m['ram_total_gb']:.0f} GB")

    h = st.session_state.gpu_hist
    if len(h["t"]) > 2:
        g1, g2 = st.columns(2)
        with g1:
            st.caption("GPU utilization %")
            st.area_chart(pd.DataFrame({"util %": list(h["util"])}), height=160)
        with g2:
            st.caption("Power draw (W)")
            st.area_chart(pd.DataFrame({"watts": list(h["power"])}), height=160)


if hasattr(st, "fragment"):
    @st.fragment(run_every="2s")
    def gpu_panel():
        render_gpu_panel()
    gpu_panel()
else:  # older streamlit — render once per interaction
    render_gpu_panel()

st.divider()

# ── Two-column body: chat | inference analytics ──────────────────────────────
col_chat, col_obs = st.columns([3, 2], gap="large")

with col_chat:
    st.subheader("Customer support chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                st.caption(
                    f"intent **{meta['intent']}** ({meta['confidence']:.0%}) · "
                    f"{meta['tokens']} tok @ **{meta['tps']:.1f} tok/s** · "
                    f"pipeline {meta['total_ms']:.0f} ms"
                )
                with st.expander("pipeline trace"):
                    guard_in = ", ".join(meta.get("guard_in", [])) or "clean"
                    guard_out = ", ".join(meta.get("guard_out", [])) or "clean"
                    st.markdown(
                        f"- guardrails: `{meta.get('guard_ms', 0):.2f} ms` · "
                        f"input: `{guard_in}` · output: `{guard_out}`\n"
                        f"- intent classification: `{meta['intent_ms']:.2f} ms`\n"
                        f"- RAG retrieval ({meta['rag_method']}): `{meta['rag_ms']:.1f} ms`\n"
                        f"- generation ({meta['model']}): `{meta['gen_ms']:.0f} ms`"
                    )
                    for c in meta["chunks"]:
                        st.markdown(f"> {c['text']}")
                if "base_response" in msg:
                    with st.expander("⚠️ base model (no fine-tuning) answered"):
                        st.markdown(msg["base_response"])
                        st.caption(msg["base_caption"])

with col_obs:
    st.subheader("Inference analytics")
    log = st.session_state.query_log
    if not log:
        st.info("Ask a question to populate per-query analytics.")
    else:
        df = pd.DataFrame(log)
        a1, a2, a3 = st.columns(3)
        a1.metric("Queries", len(df))
        a2.metric("Avg tok/s", f"{df.tps.mean():.1f}")
        a3.metric("Avg latency", f"{df.total_ms.mean()/1000:.1f}s")

        st.caption("Tokens/sec per query")
        st.bar_chart(df[["tps"]], height=160)

        st.caption("Latency breakdown per query (ms)")
        st.bar_chart(df[["intent_ms", "rag_ms", "gen_ms"]], height=180)

        st.caption("Intent distribution")
        st.bar_chart(df["intent"].value_counts(), height=140)


# ── Chat input (top level — triggers full rerun) ─────────────────────────────
if prompt := st.chat_input("Ask a telecom support question..."):
    # Conversation memory: last 2 exchanges, compact
    history = None
    if use_memory:
        turns = []
        for m in st.session_state.messages[-4:]:
            who = "Customer" if m["role"] == "user" else "Agent"
            turns.append(f"{who}: {m['content'][:200]}")
        history = " | ".join(turns) if turns else None

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Running pipeline..."):
        r = st.session_state.orchestrator.process(prompt, history=history)

    base_fields = {}
    if compare_base and "base_generator" in st.session_state:
        with st.spinner("Asking the base model the same question..."):
            bg = st.session_state.base_generator.generate(
                prompt, r.retrieval.chunks, r.intent.intent)
        base_fields = {
            "base_response": bg.response,
            "base_caption": f"{bg.tokens_generated} tok @ {bg.tokens_per_second:.1f} tok/s · "
                            f"{bg.inference_time_ms:.0f} ms",
        }

    st.session_state.messages.append({
        "role": "assistant",
        "content": r.generation.response,
        **base_fields,
        "meta": {
            "intent": r.intent.intent,
            "confidence": r.intent.confidence,
            "tokens": r.generation.tokens_generated,
            "tps": r.generation.tokens_per_second,
            "total_ms": r.total_pipeline_ms,
            "intent_ms": r.intent_ms,
            "rag_ms": r.rag_ms,
            "gen_ms": r.generation_ms,
            "rag_method": r.retrieval.retrieval_method,
            "model": r.generation.model_used,
            "chunks": r.retrieval.chunks,
            "guard_ms": r.guardrail_ms,
            "guard_in": r.guardrail_input_flags,
            "guard_out": r.guardrail_output_flags,
        },
    })
    st.session_state.query_log.append({
        "intent": r.intent.intent,
        "tps": r.generation.tokens_per_second,
        "total_ms": r.total_pipeline_ms,
        "intent_ms": r.intent_ms,
        "rag_ms": r.rag_ms,
        "gen_ms": r.generation_ms,
    })
    st.rerun()
