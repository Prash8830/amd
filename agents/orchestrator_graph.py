"""LangGraph orchestration backend.

The same eight-stage pipeline as the classic orchestrator, expressed as a
LangGraph StateGraph: every agent is a node, and the three early exits
(guardrails block, clarity gate, cache hit) plus the fast/expert split are
conditional edges. Behaviour is identical — it reuses the orchestrator's own
agent instances and the shared PipelineResult / _early_result helpers — but
the control flow is now a declarative graph you can inspect and render.

Activate with USE_LANGGRAPH=1. Bonus: pipeline_mermaid() returns the graph's
own Mermaid source — an architecture diagram generated from the executing graph.
"""

from __future__ import annotations
import time
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.guardrails import BLOCKED_RESPONSE
from agents.mcp_client import call_mcp_tool
from agents.orchestrator import PipelineResult, _early_result, TRUST_THRESHOLD


class PipeState(TypedDict, total=False):
    query: str
    history: Optional[str]
    t0: float
    safe_query: str
    gin_flags: list
    cres: Any
    intent: Any
    routing_text: str
    t_pre: float
    t1: float
    t2: float
    t3: float
    retrieval: Any
    mcp_tools: list
    route: str
    route_reason: str
    generation: Any
    gout_flags: list
    trust: float
    escalated: bool
    done: bool
    result: Any


def build_pipeline_graph(orch):
    """Compile the StateGraph, closing over the orchestrator's agents."""

    def guardrails_in(s: PipeState) -> PipeState:
        t0 = time.perf_counter()
        gin = orch.guardrails.check_input(s["query"])
        if gin.blocked:
            res = _early_result(gin.masked_query, BLOCKED_RESPONSE,
                                "guardrails (blocked before model)", t0,
                                gin.flags, intent_name="blocked", blocked=True)
            return {"t0": t0, "done": True, "result": res}
        return {"t0": t0, "safe_query": gin.masked_query, "gin_flags": gin.flags}

    def clarity(s: PipeState) -> PipeState:
        cl = orch.clarity.check(s["safe_query"], s.get("history"))
        if not cl.clear:
            res = _early_result(s["safe_query"], cl.clarifying_question,
                                f"clarity agent ({cl.reason})", s["t0"], s["gin_flags"],
                                intent_name="needs clarification", clarification=True)
            return {"done": True, "result": res}
        return {}

    def cache(s: PipeState) -> PipeState:
        cres = orch.cache.lookup(s["safe_query"])
        if cres.hit:
            res = _early_result(
                s["safe_query"], cres.answer, "semantic cache (human-approved)",
                s["t0"], s["gin_flags"], route="cache",
                intent_name=orch.classifier.classify(s["safe_query"]).intent,
                route_reason=f"ground-truth hit, similarity {cres.similarity:.2f}")
            return {"cres": cres, "done": True, "result": res}
        return {"cres": cres, "t_pre": time.perf_counter()}

    def intent(s: PipeState) -> PipeState:
        ir = orch.classifier.classify(s["safe_query"])
        routing_text = s["safe_query"]
        if s.get("history") and not ir.matched_keywords:
            routing_text = f"{s['history']} {s['safe_query']}"
            ir = orch.classifier.classify(routing_text)
        return {"intent": ir, "routing_text": routing_text, "t1": time.perf_counter()}

    def evidence(s: PipeState) -> PipeState:
        ir, cres = s["intent"], s["cres"]
        rr = orch.rag.retrieve(s["routing_text"], ir.intent, top_k=3)
        mcp_tools: list = []
        if orch.mcp_on and ir.intent == "network":
            outage = call_mcp_tool("get_outage_status", {"area": "customer-area"})
            if outage:
                mcp_tools.append("get_outage_status")
                status = (f"OUTAGE ACTIVE ({outage.get('class')}), estimated restore "
                          f"{outage.get('eta_restore')}" if outage.get("outage")
                          else "no outage reported in the customer's area")
                rr.chunks.append({"id": "mcp_outage", "score": 1.0,
                                  "text": f"Live network status (via MCP, {outage.get('checked_at', '')[:16]}): {status}."})
        if cres.assist:
            rr.chunks.append({"id": "cache_assist", "score": cres.similarity,
                              "text": f"Approved answer to a similar past question ('{cres.question}'): {cres.answer}"})
        return {"retrieval": rr, "mcp_tools": mcp_tools, "t2": time.perf_counter()}

    def route(s: PipeState) -> PipeState:
        r, reason = "expert", "single-model deployment"
        if orch.fast_generator is not None:
            d = orch.router.route(s["safe_query"], s["intent"].confidence)
            r, reason = d.route, d.reason
        return {"route": r, "route_reason": reason}

    def generate(s: PipeState) -> PipeState:
        agent = (orch.fast_generator if s["route"] == "fast" and orch.fast_generator
                 else orch.generator)
        gr = agent.generate(s["safe_query"], s["retrieval"].chunks,
                            s["intent"].intent, history=s.get("history"))
        gout = orch.guardrails.check_output(gr.response, s["retrieval"].chunks)
        gr.response = gout.response
        return {"generation": gr, "gout_flags": gout.flags, "t3": time.perf_counter()}

    def trust_gate(s: PipeState) -> PipeState:
        ir, rr, flags = s["intent"], s["retrieval"], s["gout_flags"]
        trust = 1.0
        trust -= 0.45 * sum(1 for f in flags if f.startswith("unverified_amount"))
        trust -= 0.15 * sum(1 for f in flags if "pii_leak" in f)
        if ir.confidence < 0.5:
            trust -= 0.2
        if not rr.chunks:
            trust -= 0.1
        trust = max(0.0, round(trust, 2))
        escalated = trust < TRUST_THRESHOLD

        mcp_tools = list(s.get("mcp_tools", []))
        expert = ticket = None
        if escalated and orch.mcp_on:
            expert = call_mcp_tool("find_expert", {"domain": ir.intent})
            if expert:
                mcp_tools.append("find_expert")
            ticket = call_mcp_tool("create_ticket", {
                "summary": f"Low-trust answer ({trust:.2f}) needs review — \"{s['safe_query'][:120]}\"",
                "domain": ir.intent, "assignee": (expert or {}).get("name", ""),
                "severity": "P2" if trust < 0.35 else "P3"})
            if ticket:
                mcp_tools.append("create_ticket")
        t4 = time.perf_counter()

        res = PipelineResult(
            query=s["safe_query"], intent=ir, retrieval=rr, generation=s["generation"],
            total_pipeline_ms=round((t4 - s["t0"]) * 1000, 1),
            intent_ms=round((s["t1"] - s["t_pre"]) * 1000, 2),
            rag_ms=round((s["t2"] - s["t1"]) * 1000, 1),
            generation_ms=round((s["t3"] - s["t2"]) * 1000, 1),
            guardrail_ms=round(((s["t_pre"] - s["t0"]) + (t4 - s["t3"])) * 1000, 2),
            guardrail_input_flags=s["gin_flags"], guardrail_output_flags=flags,
            route=s["route"], route_reason=s["route_reason"],
            trust_score=trust, escalated=escalated,
            escalation_expert=expert, ticket=ticket, mcp_tools_used=mcp_tools)
        return {"trust": trust, "escalated": escalated, "result": res}

    g = StateGraph(PipeState)
    for name, fn in [("guardrails_in", guardrails_in), ("clarity", clarity),
                     ("cache", cache), ("intent", intent), ("evidence", evidence),
                     ("route", route), ("generate", generate), ("trust_gate", trust_gate)]:
        g.add_node(name, fn)

    g.set_entry_point("guardrails_in")
    _gate = lambda nxt: (lambda s: END if s.get("done") else nxt)
    g.add_conditional_edges("guardrails_in", _gate("clarity"), {END: END, "clarity": "clarity"})
    g.add_conditional_edges("clarity", _gate("cache"), {END: END, "cache": "cache"})
    g.add_conditional_edges("cache", _gate("intent"), {END: END, "intent": "intent"})
    g.add_edge("intent", "evidence")
    g.add_edge("evidence", "route")
    g.add_edge("route", "generate")
    g.add_edge("generate", "trust_gate")
    g.add_edge("trust_gate", END)
    return g.compile()


def run_pipeline(graph, query: str, history: Optional[str] = None) -> PipelineResult:
    final = graph.invoke({"query": query, "history": history})
    return final["result"]


def pipeline_mermaid(orch) -> str:
    """Mermaid source of the compiled graph — a diagram from the live graph."""
    return build_pipeline_graph(orch).get_graph().draw_mermaid()
