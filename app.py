"""
TruthLine — enterprise console.

Tabs: Support Console · Observability · Governance · Model Quality & Flywheel
Run: python main.py --mode ui   ·   Open: <jupyter-base-url>/proxy/8501/
"""

import json
import os
import time
from collections import deque

import pandas as pd
import streamlit as st

from config import BASE_MODEL_ID, PRODUCT_NAME, PRODUCT_TAGLINE

st.set_page_config(page_title=f"{PRODUCT_NAME} — AMD ROCm", page_icon="📡", layout="wide")

from utils.amd_metrics import get_system_metrics

HISTORY = 120  # GPU history ticks (~4 min at 2s)


# ── Session state ─────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("query_log", [])
if "gpu_hist" not in ss:
    ss.gpu_hist = {k: deque(maxlen=HISTORY) for k in ("t", "util", "vram", "power", "temp")}

if "orchestrator" not in ss:
    with st.spinner(f"Loading {BASE_MODEL_ID} to GPU — first load takes a minute..."):
        from agents.orchestrator import TelecomOrchestrator
        ss.orchestrator = TelecomOrchestrator()


def _sample_gpu():
    m = get_system_metrics()
    g = m["gpu"]
    h = ss.gpu_hist
    h["t"].append(time.strftime("%H:%M:%S"))
    h["util"].append(g.gpu_utilization_pct)
    h["vram"].append(g.vram_used_mb / 1024 if g.vram_used_mb else 0)
    h["power"].append(g.power_draw_w)
    h["temp"].append(g.gpu_temp_c)
    return m


# ── Sidebar: product + demo controls ─────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 📡 {PRODUCT_NAME}")
    st.caption(PRODUCT_TAGLINE)
    st.divider()
    st.header("Demo controls")
    compare_base = st.toggle(
        "Hallucination check: also ask the base model",
        value=False,
        help="Loads a second, un-fine-tuned copy of the model (~28 GB VRAM) and "
             "answers every query with both. Ask about internal codes to watch "
             "the base model hallucinate while the fine-tuned one answers correctly.",
    )
    use_memory = st.toggle(
        "Conversation memory",
        value=True,
        help="Carries recent turns into routing, retrieval, and the prompt so "
             "follow-ups work: 'My router is an HG-2410' → 'the light is red'.",
    )
    if st.button("Clear conversation"):
        ss.messages = []
        st.rerun()

    router_on = ss.orchestrator.fast_generator is not None
    st.caption(("🔀 Model router: **active** (1.5B fast + 14B expert)" if router_on
                else "Model router: single-model mode — train the fast adapter with "
                     "`BASE_MODEL_ID=Qwen/Qwen2.5-1.5B python main.py --mode finetune`"))
    st.divider()
    st.caption("**Try:** `What does billing code B-204 mean?` · "
               "`My HG-2410 LOS light is red` · `eSIM fails with ERR-2077` · "
               "`it's not working` (clarity gate) · "
               "`My number is 555-867-5309 and my net is slow` (PII masking)")

if compare_base and "base_generator" not in ss:
    with st.spinner("Loading base model copy for comparison (~1 min)..."):
        from agents.response_generator import ResponseGeneratorAgent
        ss.base_generator = ResponseGeneratorAgent(use_adapter=False)


# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"📡 {PRODUCT_NAME}")
st.caption(
    f"{PRODUCT_TAGLINE} · **{BASE_MODEL_ID}** (LoRA, merged) · "
    "pipeline: guardrails → clarity → cache → intent → RAG → router → generation → trust gate"
)

tab_chat, tab_obs, tab_gov, tab_quality = st.tabs(
    ["💬 Support console", "📊 Observability", "🛡️ Governance", "🎯 Model quality & flywheel"])


# ══ TAB 1 — Support console ═══════════════════════════════════════════════════
with tab_chat:
    if not ss.messages:
        st.info("Ask a question below — every answer carries its full pipeline trace. "
                "Suggested demo queries are in the sidebar.")

    for i, msg in enumerate(ss.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                trust_badge = ("🟢" if meta.get("trust", 1) >= 0.8 else
                               "🟡" if meta.get("trust", 1) >= 0.6 else "🔴 escalated")
                st.caption(
                    f"intent **{meta['intent']}** ({meta['confidence']:.0%}) · "
                    f"route **{meta.get('route', 'expert')}** · "
                    f"trust {meta.get('trust', 1.0):.2f} {trust_badge} · "
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
                        f"- model router: `{meta.get('route', 'expert')}` — {meta.get('route_reason', '')}\n"
                        f"- generation ({meta['model']}): `{meta['gen_ms']:.0f} ms`\n"
                        f"- trust score: `{meta.get('trust', 1.0):.2f}` "
                        f"{'→ **escalated to human review**' if meta.get('escalated') else '(served)'}"
                    )
                    for c in meta["chunks"]:
                        st.markdown(f"> {c['text']}")
                if "base_response" in msg:
                    with st.expander("⚠️ base model (no fine-tuning) answered"):
                        st.markdown(msg["base_response"])
                        st.caption(msg["base_caption"])

                if msg.get("fb"):
                    st.caption("feedback recorded: " +
                               ("👍 → training set + semantic cache" if msg["fb"] == "approved"
                                else "👎 → review queue"))
                else:
                    fb1, fb2, _ = st.columns([1, 1, 8])
                    user_q = next((m["content"] for m in reversed(ss.messages[:i])
                                   if m["role"] == "user"), "")
                    if fb1.button("👍", key=f"up_{i}"):
                        from data.feedback_store import append_feedback
                        append_feedback(user_q, msg["content"], "approved")
                        msg["fb"] = "approved"
                        ss.orchestrator.cache.refresh()
                        st.rerun()
                    if fb2.button("👎", key=f"down_{i}"):
                        from data.feedback_store import append_feedback
                        append_feedback(user_q, msg["content"], "rejected")
                        msg["fb"] = "rejected"
                        ss.orchestrator.cache.refresh()
                        st.rerun()


# ══ TAB 2 — Observability ═════════════════════════════════════════════════════
def render_observability():
    m = _sample_gpu()
    g = m["gpu"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("GPU util", f"{g.gpu_utilization_pct:.0f}%" if g.available else "N/A")
    c2.metric("VRAM", f"{g.vram_used_mb/1024:.0f}/{g.vram_total_mb/1024:.0f} GB" if g.available else "N/A")
    c3.metric("GPU temp", f"{g.gpu_temp_c:.0f}°C" if g.available else "N/A")
    c4.metric("Power", f"{g.power_draw_w:.0f} W" if g.available else "N/A")
    c5.metric("CPU", f"{m['cpu_pct']:.0f}%")
    c6.metric("RAM", f"{m['ram_used_gb']:.0f}/{m['ram_total_gb']:.0f} GB")

    h = ss.gpu_hist
    if len(h["t"]) > 2:
        g1, g2 = st.columns(2)
        with g1:
            st.caption("GPU utilization % (rocm-smi, 2s ticks)")
            st.area_chart(pd.DataFrame({"util %": list(h["util"])}), height=170)
        with g2:
            st.caption("Power draw (W)")
            st.area_chart(pd.DataFrame({"watts": list(h["power"])}), height=170)


with tab_obs:
    st.subheader("AMD MI300X telemetry — live via rocm-smi")
    if hasattr(st, "fragment"):
        @st.fragment(run_every="2s")
        def gpu_panel():
            render_observability()
        gpu_panel()
    else:
        render_observability()

    st.divider()
    st.subheader("Inference analytics (this session)")
    log = ss.query_log
    if not log:
        st.caption("Ask a question in the support console to populate analytics.")
    else:
        df = pd.DataFrame(log)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Queries", len(df))
        a2.metric("Avg tok/s", f"{df.tps.mean():.1f}")
        a3.metric("Avg latency", f"{df.total_ms.mean()/1000:.1f}s")
        zero_gpu = int((df.get("route", pd.Series(dtype=str)) == "cache").sum())
        a4.metric("Zero-GPU answers", zero_gpu)

        b1, b2 = st.columns(2)
        with b1:
            st.caption("Latency breakdown per query (ms)")
            st.bar_chart(df[["intent_ms", "rag_ms", "gen_ms"]], height=190)
        with b2:
            st.caption("Serving tier per query (cost right-sizing)")
            if "route" in df:
                st.bar_chart(df["route"].value_counts(), height=190)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Tokens/sec per query")
            st.bar_chart(df[["tps"]], height=170)
        with c2:
            st.caption("Intent distribution")
            st.bar_chart(df["intent"].value_counts(), height=170)


# ══ TAB 3 — Governance ════════════════════════════════════════════════════════
with tab_gov:
    g1, g2 = st.columns(2)

    with g1:
        escalated = [m for m in ss.messages
                     if m.get("role") == "assistant" and m.get("meta", {}).get("escalated")]
        st.subheader(f"🔴 Human review queue ({len(escalated)})")
        if not escalated:
            st.caption("No answers below the trust threshold (0.6).")
        for e in escalated:
            st.warning(f"trust {e['meta']['trust']:.2f} — {e['content'][:180]}...")

        if ss.query_log:
            df = pd.DataFrame(ss.query_log)
            if "trust" in df:
                st.caption("Trust score per query (threshold 0.6)")
                st.line_chart(df[["trust"]], height=170)

    with g2:
        st.subheader("🛡️ Guardrail activity")
        events = []
        for m in ss.messages:
            meta = m.get("meta", {})
            for f in meta.get("guard_in", []):
                events.append({"stage": "input", "event": f})
            for f in meta.get("guard_out", []):
                events.append({"stage": "output", "event": f})
        if not events:
            st.caption("No PII, injection, or unverified-claim events yet. "
                       "Try the PII demo query in the sidebar.")
        else:
            st.dataframe(pd.DataFrame(events), use_container_width=True, height=240)
        st.caption(
            "Input: PII masked + injections blocked **before** the model or logs "
            "see them. Output: PII leaks redacted; dollar amounts without grounding "
            "are flagged as `unverified_amount`."
        )


# ══ TAB 4 — Model quality & flywheel ══════════════════════════════════════════
with tab_quality:
    q1, q2 = st.columns(2)

    with q1:
        st.subheader("🎯 Measured accuracy — base vs fine-tuned")
        if os.path.exists("eval_results.json"):
            with open("eval_results.json") as f:
                ev = json.load(f)
            cats = {k: v for k, v in ev["summary"].items() if k != "TOTAL"}
            df_acc = pd.DataFrame({
                "base": {k: v["base_pct"] for k, v in cats.items()},
                "fine-tuned": {k: v["ft_pct"] for k, v in cats.items()},
            })
            st.bar_chart(df_acc, height=240)
            tot = ev["summary"]["TOTAL"]
            e1, e2, e3 = st.columns(3)
            e1.metric("Total accuracy", f"{tot['ft_pct']}%", delta=f"+{tot['ft_pct']-tot['base_pct']} vs base")
            eff = ev.get("efficiency", {})
            if eff:
                e2.metric("Tokens/answer", f"{eff['ft_tokens']//tot['n']}",
                          delta=f"-{100 - 100*eff['ft_tokens']//max(eff['base_tokens'],1)}%",
                          delta_color="inverse")
                e3.metric("Eval wall time", f"{eff['ft_seconds']:.0f}s",
                          delta=f"vs base {eff['base_seconds']:.0f}s", delta_color="off")
            if eff:
                st.caption("Tokens per answer (same questions, same grounding)")
                st.bar_chart(pd.DataFrame({
                    "tokens/answer": {
                        "base": eff["base_tokens"] // max(tot["n"], 1),
                        "fine-tuned": eff["ft_tokens"] // max(tot["n"], 1),
                    }}), height=170)
            st.caption(f"18 held-out questions · internal codes are synthetic — a base "
                       f"model cannot know them · model: {ev.get('model', '')}")
        else:
            st.caption("Run `python evaluate.py` to populate measured accuracy.")

    with q2:
        st.subheader("🔄 Data flywheel")
        from data.feedback_store import load_feedback
        fb_all = load_feedback()
        approved_n = sum(1 for f in fb_all if f.get("label") == "approved")
        cache_n = ss.orchestrator.cache.size()

        f1, f2, f3 = st.columns(3)
        f1.metric("Approved pairs", approved_n)
        f2.metric("In review", len(fb_all) - approved_n)
        f3.metric("Cache entries", cache_n)

        st.markdown(
            "1. 👍 on any answer → pair lands in the **ground-truth DB**\n"
            "2. **Immediately**: semantic cache serves repeat questions — zero GPU\n"
            "3. **Next retrain**: `finetune.py` auto-merges approved pairs into "
            "the training set (~1 min on MI300X)\n"
            "4. 👎 → review queue for the knowledge team"
        )
        if approved_n:
            with st.expander("approved pairs (next training set)"):
                for row in fb_all:
                    if row.get("label") == "approved":
                        st.markdown(f"**Q:** {row['question'][:120]}\n\n**A:** {row['answer'][:200]}...")


# ── Chat input (top level — pinned, works from any tab) ──────────────────────
if prompt := st.chat_input(f"Ask {PRODUCT_NAME} a telecom support question..."):
    history = None
    if use_memory:
        turns = []
        for m in ss.messages[-4:]:
            who = "Customer" if m["role"] == "user" else "Agent"
            turns.append(f"{who}: {m['content'][:200]}")
        history = " | ".join(turns) if turns else None

    ss.messages.append({"role": "user", "content": prompt})
    with st.spinner("Running pipeline..."):
        r = ss.orchestrator.process(prompt, history=history)

    base_fields = {}
    if compare_base and "base_generator" in ss:
        with st.spinner("Asking the base model the same question..."):
            bg = ss.base_generator.generate(prompt, r.retrieval.chunks, r.intent.intent)
        base_fields = {
            "base_response": bg.response,
            "base_caption": f"{bg.tokens_generated} tok @ {bg.tokens_per_second:.1f} tok/s · "
                            f"{bg.inference_time_ms:.0f} ms",
        }

    ss.messages.append({
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
            "route": r.route,
            "route_reason": r.route_reason,
            "trust": r.trust_score,
            "escalated": r.escalated,
        },
    })
    ss.query_log.append({
        "intent": r.intent.intent,
        "tps": r.generation.tokens_per_second,
        "total_ms": r.total_pipeline_ms,
        "intent_ms": r.intent_ms,
        "rag_ms": r.rag_ms,
        "gen_ms": r.generation_ms,
        "route": r.route,
        "trust": r.trust_score,
    })
    # GPU sample per query keeps charts moving even without the fragment
    _sample_gpu()
    st.rerun()
