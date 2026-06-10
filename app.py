"""
Streamlit Observability Dashboard — AMD Hackathon
Shows AMD GPU metrics, inference stats, and chat interface.

Run: streamlit run app.py
"""

import time
import threading
import queue
from collections import deque

import streamlit as st

st.set_page_config(
    page_title="Telecom Chatbot — AMD ROCm",
    page_icon="📡",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = {
        "time": deque(maxlen=60),
        "gpu_util": deque(maxlen=60),
        "vram_used": deque(maxlen=60),
        "tps": deque(maxlen=60),
        "latency": deque(maxlen=60),
    }
if "orchestrator" not in st.session_state:
    with st.spinner("Loading Qwen3-14B model..."):
        from agents.orchestrator import TelecomOrchestrator
        st.session_state.orchestrator = TelecomOrchestrator()

from utils.amd_metrics import get_system_metrics

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Telecom Support Chatbot")
st.caption("Powered by Qwen3-14B QLoRA · AMD ROCm · Multi-Agent Pipeline")

# ── Layout ────────────────────────────────────────────────────────────────────
col_chat, col_metrics = st.columns([3, 2])

# ── Metrics panel ─────────────────────────────────────────────────────────────
with col_metrics:
    st.subheader("AMD Hardware Metrics")

    metrics = get_system_metrics()
    gpu = metrics["gpu"]

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)

    with m1:
        gpu_label = f"{gpu.gpu_utilization_pct:.0f}%" if gpu.available else "N/A"
        st.metric("GPU Utilization", gpu_label)
    with m2:
        vram_label = f"{gpu.vram_used_mb:.0f}/{gpu.vram_total_mb:.0f} MB" if gpu.available else "N/A"
        st.metric("VRAM Used", vram_label)
    with m3:
        temp_label = f"{gpu.gpu_temp_c:.0f}°C" if gpu.available else "N/A"
        st.metric("GPU Temp", temp_label)
    with m4:
        power_label = f"{gpu.power_draw_w:.0f}W" if gpu.available else "N/A"
        st.metric("Power Draw", power_label)

    st.divider()

    m5, m6 = st.columns(2)
    with m5:
        st.metric("CPU", f"{metrics['cpu_pct']:.0f}%")
    with m6:
        st.metric("RAM", f"{metrics['ram_used_gb']}/{metrics['ram_total_gb']} GB")

    # Recent inference stats
    if st.session_state.messages:
        last = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
        if last and "stats" in last:
            st.divider()
            st.subheader("Last Inference Stats")
            s = last["stats"]
            st.metric("Tokens/sec", f"{s['tps']:.1f}")
            st.metric("Inference Time", f"{s['inference_ms']:.0f}ms")
            st.metric("Pipeline Latency", f"{s['pipeline_ms']:.0f}ms")
            st.metric("Model", s["model"])

    # Charts
    hist = st.session_state.metrics_history
    if len(hist["time"]) > 1:
        st.divider()
        st.subheader("GPU Utilization Over Time")
        import pandas as pd
        df_gpu = pd.DataFrame({"GPU %": list(hist["gpu_util"])})
        st.line_chart(df_gpu)

        if any(v > 0 for v in hist["tps"]):
            st.subheader("Tokens/sec Over Time")
            df_tps = pd.DataFrame({"tok/s": list(hist["tps"])})
            st.line_chart(df_tps)

# ── Chat panel ────────────────────────────────────────────────────────────────
with col_chat:
    st.subheader("Customer Support Chat")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "meta" in msg:
                    meta = msg["meta"]
                    st.caption(
                        f"Intent: **{meta['intent']}** ({meta['confidence']:.0%}) · "
                        f"RAG: {meta['retrieval']} · "
                        f"{meta['tps']:.1f} tok/s · {meta['pipeline_ms']:.0f}ms"
                    )

    if prompt := st.chat_input("Ask a telecom support question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Processing..."):
            result = st.session_state.orchestrator.process(prompt)

        response_text = result.generation.response
        meta = {
            "intent": result.intent.intent,
            "confidence": result.intent.confidence,
            "retrieval": result.retrieval.retrieval_method,
            "tps": result.generation.tokens_per_second,
            "pipeline_ms": result.total_pipeline_ms,
        }
        stats = {
            "tps": result.generation.tokens_per_second,
            "inference_ms": result.generation.inference_time_ms,
            "pipeline_ms": result.total_pipeline_ms,
            "model": result.generation.model_used,
        }

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "meta": meta,
            "stats": stats,
        })

        # Update metrics history
        now = time.time()
        hist["time"].append(now)
        hist["gpu_util"].append(gpu.gpu_utilization_pct)
        hist["vram_used"].append(gpu.vram_used_mb)
        hist["tps"].append(result.generation.tokens_per_second)
        hist["latency"].append(result.total_pipeline_ms)

        st.rerun()
