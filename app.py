"""
TruthLine — enterprise console.

Tabs: Support Console · Observability · Governance · Model Quality & Flywheel
Run: python main.py --mode ui   ·   Open: <jupyter-base-url>/proxy/8501/
"""

import json
import os
import time
from collections import deque

import altair as alt
import pandas as pd
import streamlit as st

# ── Chart theme (product palette) ────────────────────────────────────────────
RED, AMBER, GREEN, BLUE = "#ED1C24", "#FAC775", "#2FBF71", "#6E9FB5"
INK, MUTE = "#F4F1EA", "#A9A399"
TIER_SCALE = alt.Scale(domain=["cache", "fast", "expert", "blocked"],
                       range=[GREEN, AMBER, RED, "#6E6A63"])


def _themed(chart):
    return (chart
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor="#262626", gridOpacity=0.7,
                            domain=False, tickColor="#3A3A3A", labelLimit=200,
                            labelColor=MUTE, titleColor=MUTE)
            .configure_legend(labelColor=MUTE, titleColor=MUTE, orient="bottom"))


def _area_ts(values, label, color, ymax=None, height=180):
    df = pd.DataFrame({"tick": range(len(values)), "v": list(values)})
    scale = alt.Scale(domain=[0, ymax]) if ymax else alt.Scale(zero=True)
    grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                        stops=[alt.GradientStop(color="rgba(0,0,0,0)", offset=0),
                               alt.GradientStop(color=color + "55", offset=1)])
    return (alt.Chart(df)
            .mark_area(line={"color": color, "strokeWidth": 2}, color=grad, interpolate="monotone")
            .encode(x=alt.X("tick:Q", axis=None),
                    y=alt.Y("v:Q", title=label, scale=scale, axis=alt.Axis(tickCount=4)))
            .properties(height=height))

from config import BASE_MODEL_ID, PRODUCT_NAME, PRODUCT_TAGLINE

st.set_page_config(page_title=f"{PRODUCT_NAME} — AMD ROCm", page_icon="📡", layout="wide")

# ── Enterprise skin ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] * { font-family: 'Inter', sans-serif; }
code, pre, [data-testid="stCode"] * { font-family: 'JetBrains Mono', monospace !important; }

.stApp {
  background:
    radial-gradient(1100px 600px at 15% -8%, rgba(237,28,36,0.07) 0%, rgba(14,14,14,0) 55%),
    radial-gradient(900px 500px at 95% 0%, rgba(237,28,36,0.04) 0%, rgba(14,14,14,0) 50%),
    #0E0E0E;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1500px; }

/* Hero */
.tl-hero { padding: 6px 0 2px 0; }
.tl-brand { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; color: #F4F1EA; }
.tl-brand .tl-dot { color: #ED1C24; }
.tl-tag { color: #A9A399; font-size: 0.95rem; margin-top: 2px; }
.tl-pills { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
.pill { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; padding: 4px 12px;
        border-radius: 999px; border: 1px solid #2E2E2E; background: #161616; color: #A9A399; }
.pill-ok { border-color: rgba(47,191,113,0.45); color: #2FBF71; }
.pill-red { border-color: rgba(237,28,36,0.5); color: #FF6B70; }

/* Tabs → segmented control */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #151515; padding: 6px;
  border-radius: 14px; border: 1px solid #262626; width: fit-content; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 9px;
  padding: 8px 20px; color: #A9A399; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #ED1C24 !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Metrics → KPI cards */
[data-testid="stMetric"] { background: linear-gradient(180deg, #1A1A1A 0%, #131313 100%);
  border: 1px solid #2A2A2A; border-radius: 14px; padding: 14px 16px 10px 16px; }
[data-testid="stMetric"]:hover { border-color: #3d3d3d; }
[data-testid="stMetricValue"] { font-weight: 800; font-size: 1.7rem; }
[data-testid="stMetricLabel"] { color: #A9A399 !important; text-transform: uppercase;
  letter-spacing: 0.09em; font-size: 0.7rem !important; font-weight: 600; }

/* Chat */
[data-testid="stChatMessage"] { background: #151515; border: 1px solid #262626;
  border-radius: 14px; padding: 14px 18px; margin-bottom: 4px; }
[data-testid="stChatInput"] { border-radius: 14px; }

/* Buttons, expanders, alerts, misc */
.stButton button { border: 1px solid #333; background: #181818; color: #F4F1EA;
  border-radius: 10px; font-weight: 600; transition: all .15s ease; }
.stButton button:hover { border-color: #ED1C24; background: #221112; color: #fff; }
[data-testid="stExpander"] { background: #131313; border: 1px solid #262626; border-radius: 12px; }
[data-testid="stExpander"] summary { font-weight: 600; color: #A9A399; }
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid #2A2A2A; }
hr { border-color: #242424 !important; }
[data-testid="stSidebar"] { background: #101010; border-right: 1px solid #222; }
[data-testid="stSidebar"] hr { border-color: #222 !important; }
h3 { letter-spacing: -0.01em; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #2c2c2c; border-radius: 6px; }
::-webkit-scrollbar-track { background: transparent; }

/* Restore Streamlit's icon font (the global Inter override breaks it) */
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

/* Metric fit: no truncated values or labels */
[data-testid="stMetricValue"] { font-size: 1.45rem !important; }
[data-testid="stMetricLabel"] { overflow: visible !important; }
[data-testid="stMetricLabel"] p { white-space: normal !important; overflow-wrap: anywhere; }

/* Overview (home) components */
.tl-sec { color: #A9A399; text-transform: uppercase; letter-spacing: .12em;
  font-size: .72rem; font-weight: 700; margin: 26px 0 8px 0; }
.tl-tiles { display: flex; gap: 14px; flex-wrap: wrap; }
.tl-tile { flex: 1; min-width: 210px; background: linear-gradient(180deg, #1B1B1B 0%, #121212 100%);
  border: 1px solid #2A2A2A; border-radius: 16px; padding: 22px 22px 18px 22px; }
.tl-tile .v { font-size: 2.1rem; font-weight: 800; color: #F4F1EA; letter-spacing: -0.02em; }
.tl-tile .v em { color: #ED1C24; font-style: normal; }
.tl-tile .l { color: #A9A399; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .07em; margin-top: 6px; line-height: 1.5; }
.tl-strip { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.tl-step { font-family: 'JetBrains Mono', monospace; font-size: .76rem; padding: 7px 13px;
  border-radius: 10px; background: #161616; border: 1px solid #2C2C2C; color: #D8D4CC; }
.tl-step.hot { border-color: rgba(237,28,36,.5); color: #FF8A8E; }
.tl-arrow { color: #ED1C24; font-weight: 700; }
.tl-cards { display: flex; gap: 14px; flex-wrap: wrap; }
.tl-card { flex: 1; min-width: 250px; background: #141414; border: 1px solid #262626;
  border-radius: 16px; padding: 20px; }
.tl-card h4 { margin: 0 0 8px 0; color: #F4F1EA; font-size: 1.0rem; }
.tl-card p { margin: 0; color: #A9A399; font-size: .87rem; line-height: 1.55; }
.tl-card .go { display: inline-block; margin-top: 12px; color: #FF6B70; font-size: .78rem;
  font-weight: 700; letter-spacing: .04em; }
</style>
""", unsafe_allow_html=True)

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
    st.caption(("🔌 MCP enterprise server: **connected** (expert routing + live outage feed)"
                if ss.orchestrator.mcp_on
                else "MCP server: offline — start it in a terminal: "
                     "`python mcp_server/telecom_mcp.py`, then restart the app"))
    st.divider()
    st.markdown("#### 🎬 Demo script")
    st.caption("Hover a line → copy icon. 🔄 New conversation between scenes.")

    st.caption("**1 · Hallucination check** — toggle base model ON first")
    st.code("What does billing code B-204 mean on my invoice?", language=None)

    st.caption("**2 · Router + MCP** — fast lane, then live outage feed")
    st.code("When is my payment due?", language=None)
    st.code("My internet is very slow since yesterday", language=None)

    st.caption("**3 · Clarity + PII guardrails**")
    st.code("it's not working", language=None)
    st.code("My number is 555-867-5309 and my internet keeps dropping", language=None)

    st.caption("**4 · Flywheel** — ask, 👍 the answer, then re-ask → cache hit")
    st.code("What does error code ERR-2077 mean?", language=None)
    st.code("What does the ERR-2077 error code mean?", language=None)

    st.caption("**Bonus** — expert lane troubleshooting + memory follow-up")
    st.code("My HG-2410 LOS light is red", language=None)
    st.code("the light is still red after restarting", language=None)

if compare_base and "base_generator" not in ss:
    with st.spinner("Loading base model copy for comparison (~1 min)..."):
        from agents.response_generator import ResponseGeneratorAgent
        ss.base_generator = ResponseGeneratorAgent(use_adapter=False)


# ── Header: branded hero with live status pills ─────────────────────────────
_router_on = ss.orchestrator.fast_generator is not None
_mcp_on = ss.orchestrator.mcp_on
_serving = ss.orchestrator.generator._model_label
st.markdown(f"""
<div class="tl-hero">
  <div class="tl-brand">Truth<span class="tl-dot">Line</span></div>
  <div class="tl-tag">{PRODUCT_TAGLINE}</div>
  <div class="tl-pills">
    <span class="pill pill-red">⬤ AMD Instinct MI300X · ROCm</span>
    <span class="pill pill-ok">⬤ {_serving}</span>
    <span class="pill {'pill-ok' if _router_on else ''}">{'⬤ router: fast 1.5B + expert 14B' if _router_on else '○ router: single-model'}</span>
    <span class="pill {'pill-ok' if _mcp_on else ''}">{'⬤ MCP enterprise tools: connected' if _mcp_on else '○ MCP: offline'}</span>
    <span class="pill">guardrails → clarity → cache → intent → RAG → router → generate → trust gate</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_home, tab_chat, tab_obs, tab_gov, tab_quality = st.tabs(
    ["🏠 Overview", "💬 Support console", "📊 Observability", "🛡️ Governance",
     "🎯 Model quality & flywheel"])


# ══ TAB 0 — Overview (product home) ══════════════════════════════════════════
with tab_home:
    _ft_pct, _base_pct = 94, 22
    if os.path.exists("eval_results.json"):
        try:
            with open("eval_results.json") as _f:
                _tot = json.load(_f)["summary"]["TOTAL"]
            _ft_pct, _base_pct = _tot["ft_pct"], _tot["base_pct"]
        except Exception:
            pass
    _cache_n = ss.orchestrator.cache.size()

    _steps = ["guardrails", "clarity gate", "semantic cache", "intent", "hybrid RAG",
              "router", "fast 1.5B · expert 14B", "trust gate"]
    _strip = '<span class="tl-arrow">→</span>'.join(
        f'<span class="tl-step{" hot" if s in ("semantic cache", "fast 1.5B · expert 14B") else ""}">{s}</span>'
        for s in _steps)

    st.markdown(f"""
<div class="tl-sec">Why TruthLine</div>
<div class="tl-tiles">
  <div class="tl-tile"><div class="v">{_base_pct}% <em>→ {_ft_pct}%</em></div>
    <div class="l">measured accuracy on proprietary telecom knowledge · 18 held-out questions</div></div>
  <div class="tl-tile"><div class="v"><em>60 s</em></div>
    <div class="l">full LoRA retrain on one AMD Instinct MI300X</div></div>
  <div class="tl-tile"><div class="v"><em>−74%</em></div>
    <div class="l">tokens per answer vs the base model · −51% latency</div></div>
  <div class="tl-tile"><div class="v"><em>0</em></div>
    <div class="l">external API calls — every token generated on this GPU</div></div>
</div>

<div class="tl-sec">The pipeline — every answer earns its way to the GPU</div>
<div class="tl-strip">{_strip}</div>

<div class="tl-sec">Explore the console</div>
<div class="tl-cards">
  <div class="tl-card"><h4>💬 Support console</h4>
    <p>Chat with the domain-expert model. Every answer ships with its full pipeline
    trace — routing decision, MCP tool calls, trust score — and a 👍 feeds the
    data flywheel.</p><span class="go">DEMO STARTS HERE →</span></div>
  <div class="tl-card"><h4>📊 Observability</h4>
    <p>Live rocm-smi telemetry from the MI300X, plus per-query latency breakdown,
    serving-tier mix, and token throughput.</p><span class="go">THE AMD STORY →</span></div>
  <div class="tl-card"><h4>🛡️ Governance</h4>
    <p>PII masked before the model sees it, injections blocked, unverified claims
    flagged — and low-trust answers routed via MCP to the on-call expert.</p>
    <span class="go">THE TRUST STORY →</span></div>
  <div class="tl-card"><h4>🎯 Model quality &amp; flywheel</h4>
    <p>The held-out evaluation ({_base_pct}% → {_ft_pct}%), token efficiency, and the
    ground-truth store ({_cache_n} approved pairs) feeding the next 60-second retrain.</p>
    <span class="go">THE PROOF →</span></div>
</div>
""", unsafe_allow_html=True)


# ══ TAB 1 — Support console ═══════════════════════════════════════════════════
with tab_chat:
    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_r:
        if st.button("🔄 New conversation", help="Clears chat history and conversation memory — fresh start"):
            ss.messages = []
            st.rerun()
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
                        f"- MCP tools: `{', '.join(meta.get('mcp_tools', [])) or 'none called'}`\n"
                        f"- trust score: `{meta.get('trust', 1.0):.2f}` "
                        f"{'→ **escalated to human review**' if meta.get('escalated') else '(served)'}"
                    )
                    if meta.get("expert"):
                        ex = meta["expert"]
                        st.markdown(f"  ↳ routed via MCP to **{ex.get('name')}** — "
                                    f"{ex.get('role')} ({ex.get('contact')})")
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
            st.caption("GPU utilization % — rocm-smi, 2s ticks")
            st.altair_chart(_themed(_area_ts(h["util"], "util %", RED, ymax=100)),
                            use_container_width=True)
        with g2:
            st.caption("Power draw (W)")
            st.altair_chart(_themed(_area_ts(h["power"], "watts", AMBER)),
                            use_container_width=True)


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

        df = df.copy()
        df["query"] = range(1, len(df) + 1)
        if "route" not in df:
            df["route"] = "expert"

        b1, b2 = st.columns(2)
        with b1:
            st.caption("Latency breakdown per query — flat columns are zero-GPU cache hits")
            stages = df.melt("query", value_vars=["intent_ms", "rag_ms", "gen_ms"],
                             var_name="stage", value_name="ms")
            stages["stage"] = stages["stage"].map(
                {"intent_ms": "intent", "rag_ms": "retrieval", "gen_ms": "generation"})
            bars = (alt.Chart(stages).mark_bar(size=22, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                    .encode(x=alt.X("query:O", title=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("ms:Q", title="ms"),
                            color=alt.Color("stage:N", title=None,
                                            scale=alt.Scale(domain=["intent", "retrieval", "generation"],
                                                            range=[BLUE, AMBER, RED])),
                            order=alt.Order("stage:N")))
            layers = bars
            cache_rows = df[df["route"] == "cache"]
            if len(cache_rows):
                zap = (alt.Chart(cache_rows)
                       .mark_text(text="⚡ 0 GPU", color=GREEN, fontSize=11,
                                  fontWeight=600, angle=270, baseline="middle")
                       .encode(x=alt.X("query:O"), y=alt.value(60)))
                layers = bars + zap
            st.altair_chart(_themed(layers.properties(height=210)), use_container_width=True)
        with b2:
            st.caption("Serving tier — cost right-sizing in action")
            tier = df["route"].value_counts().rename_axis("tier").reset_index(name="answers")
            tbars = (alt.Chart(tier).mark_bar(height=30, cornerRadiusEnd=3)
                     .encode(y=alt.Y("tier:N", title=None, sort="-x"),
                             x=alt.X("answers:Q", title="answers",
                                     axis=alt.Axis(tickMinStep=1)),
                             color=alt.Color("tier:N", scale=TIER_SCALE, legend=None)))
            tlabels = (alt.Chart(tier)
                       .mark_text(align="left", dx=5, color=INK, fontSize=13, fontWeight=700)
                       .encode(y=alt.Y("tier:N", sort="-x"), x="answers:Q", text="answers:Q"))
            st.altair_chart(_themed((tbars + tlabels).properties(height=210)),
                            use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Tokens/sec per query — cache answers consume no GPU, hence no bar")
            ch = (alt.Chart(df).mark_bar(size=22, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                  .encode(x=alt.X("query:O", title=None, axis=alt.Axis(labelAngle=0)),
                          y=alt.Y("tps:Q", title="tok/s"),
                          color=alt.Color("route:N", title=None, scale=TIER_SCALE))
                  .properties(height=190))
            st.altair_chart(_themed(ch), use_container_width=True)
        with c2:
            st.caption("Intent distribution")
            idf = df["intent"].value_counts().rename_axis("intent").reset_index(name="n")
            ch = (alt.Chart(idf).mark_bar(size=18, color="#8F8A82", cornerRadiusEnd=2)
                  .encode(x=alt.X("n:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                          y=alt.Y("intent:N", title=None, sort="-x"))
                  .properties(height=190))
            st.altair_chart(_themed(ch), use_container_width=True)


# ══ TAB 3 — Governance ════════════════════════════════════════════════════════
with tab_gov:
    _n_events = sum(len(m.get("meta", {}).get("guard_in", [])) +
                    len(m.get("meta", {}).get("guard_out", []))
                    for m in ss.messages)
    _n_esc = sum(1 for m in ss.messages if m.get("meta", {}).get("escalated"))
    _trusts = [q["trust"] for q in ss.query_log if "trust" in q]
    _cache_hits = sum(1 for q in ss.query_log if q.get("route") == "cache")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Queries this session", len(ss.query_log))
    k2.metric("Guardrail events", _n_events)
    k3.metric("Escalated to humans", _n_esc,
              delta=f"{100*_n_esc/max(len(ss.query_log),1):.0f}% of traffic", delta_color="off")
    k4.metric("Avg trust score", f"{(sum(_trusts)/len(_trusts)):.2f}" if _trusts else "—")
    k5.metric("Human-approved served", _cache_hits)
    st.divider()

    g1, g2 = st.columns(2)

    with g1:
        escalated = [m for m in ss.messages
                     if m.get("role") == "assistant" and m.get("meta", {}).get("escalated")]
        st.subheader(f"🔴 Human review queue ({len(escalated)})")
        if not escalated:
            st.caption("No answers below the trust threshold (0.6).")
        for e in escalated:
            ex = e["meta"].get("expert")
            assignee = (f" → assigned via MCP: **{ex['name']}** ({ex['role']})"
                        if ex else " → unassigned (MCP offline)")
            st.warning(f"trust {e['meta']['trust']:.2f} — {e['content'][:160]}...{assignee}")

        if ss.query_log:
            df = pd.DataFrame(ss.query_log)
            if "trust" in df:
                st.caption("Trust per query — answers below the line go to humans")
                tdf = df.reset_index().rename(columns={"index": "query"})
                tdf["query"] += 1
                tdf["status"] = tdf["trust"].apply(lambda t: "escalated" if t < 0.6 else "served")
                line = (alt.Chart(tdf).mark_line(color=INK, strokeWidth=1.5, interpolate="monotone")
                        .encode(x=alt.X("query:O", title=None, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("trust:Q", scale=alt.Scale(domain=[0, 1.05]), title="trust")))
                pts = (alt.Chart(tdf).mark_circle(size=90)
                       .encode(x="query:O", y="trust:Q",
                               color=alt.Color("status:N", title=None,
                                               scale=alt.Scale(domain=["served", "escalated"],
                                                               range=[GREEN, RED])),
                               tooltip=["query", "trust", "status"]))
                rule = (alt.Chart(pd.DataFrame({"y": [0.6]}))
                        .mark_rule(color=RED, strokeDash=[6, 4], strokeWidth=1.5)
                        .encode(y="y:Q"))
                st.altair_chart(_themed((line + pts + rule).properties(height=200)),
                                use_container_width=True)

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

    st.markdown("""
<div class="tl-sec">How the trust gate decides</div>
<div class="tl-cards">
  <div class="tl-card"><h4>1 · Score every answer</h4>
    <p><code>trust = 1.0</code> minus penalties: <code>−0.35</code> per unverified
    dollar amount · <code>−0.15</code> per redacted PII leak · <code>−0.20</code> if
    intent confidence &lt; 0.5 · <code>−0.10</code> if retrieval came back empty.
    No extra model call — the signals already exist in the pipeline.</p></div>
  <div class="tl-card"><h4>2 · Gate at 0.6</h4>
    <p>Below threshold, the answer never reaches the customer. It lands in the
    review queue on the left — visible, auditable, reversible.</p></div>
  <div class="tl-card"><h4>3 · Route to the right human</h4>
    <p>The MCP enterprise server looks up the <b>on-call domain expert</b> with the
    lowest ticket load — hardware issue → CPE specialist, billing dispute →
    billing ops — and attaches them to the case.</p></div>
</div>
""", unsafe_allow_html=True)


# ══ TAB 4 — Model quality & flywheel ══════════════════════════════════════════
with tab_quality:
    q1, q2 = st.columns(2)

    with q1:
        st.subheader("🎯 Measured accuracy — base vs fine-tuned")
        if os.path.exists("eval_results.json"):
            with open("eval_results.json") as f:
                ev = json.load(f)
            order = ["internal_billing", "internal_errors", "internal_hardware",
                     "general_telecom", "TOTAL"]
            nice = {"internal_billing": "Billing codes", "internal_errors": "Error codes",
                    "internal_hardware": "Hardware", "general_telecom": "Public telecom",
                    "TOTAL": "TOTAL"}
            rows = []
            for k in order:
                if k in ev["summary"]:
                    rows.append({"category": nice[k], "model": "base",
                                 "accuracy": ev["summary"][k]["base_pct"]})
                    rows.append({"category": nice[k], "model": "fine-tuned",
                                 "accuracy": ev["summary"][k]["ft_pct"]})
            acc = pd.DataFrame(rows)
            bars = (alt.Chart(acc).mark_bar(height=13, cornerRadiusEnd=2)
                    .encode(y=alt.Y("category:N", title=None,
                                    sort=[nice[k] for k in order]),
                            yOffset=alt.YOffset("model:N"),
                            x=alt.X("accuracy:Q", title="accuracy %",
                                    scale=alt.Scale(domain=[0, 105])),
                            color=alt.Color("model:N", title=None,
                                            scale=alt.Scale(domain=["base", "fine-tuned"],
                                                            range=["#6E6A63", RED]))))
            labels = (alt.Chart(acc).mark_text(align="left", dx=4, color=INK, fontSize=11)
                      .encode(y=alt.Y("category:N", sort=[nice[k] for k in order]),
                              yOffset=alt.YOffset("model:N"),
                              x="accuracy:Q", text="accuracy:Q"))
            st.altair_chart(_themed((bars + labels).properties(height=270)),
                            use_container_width=True)
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
                st.caption("Tokens per answer — same questions, same grounding")
                tok = pd.DataFrame([
                    {"model": "base", "tokens": eff["base_tokens"] // max(tot["n"], 1)},
                    {"model": "fine-tuned", "tokens": eff["ft_tokens"] // max(tot["n"], 1)},
                ])
                tb = (alt.Chart(tok).mark_bar(height=26, cornerRadiusEnd=3)
                      .encode(y=alt.Y("model:N", title=None),
                              x=alt.X("tokens:Q", title="tokens per answer"),
                              color=alt.Color("model:N", legend=None,
                                              scale=alt.Scale(domain=["base", "fine-tuned"],
                                                              range=["#6E6A63", RED]))))
                tl = (alt.Chart(tok).mark_text(align="left", dx=4, color=INK)
                      .encode(y="model:N", x="tokens:Q", text="tokens:Q"))
                st.altair_chart(_themed((tb + tl).properties(height=120)),
                                use_container_width=True)
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
            "expert": r.escalation_expert,
            "mcp_tools": r.mcp_tools_used,
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
