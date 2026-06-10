"""Multi-agent orchestrator.

Pipeline: guardrails → clarity → intent → RAG → router → generation → guardrails
Post-generation, a trust score gates escalation to human review.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path

from agents.guardrails import GuardrailsAgent, BLOCKED_RESPONSE
from agents.clarity import ClarityAgent
from agents.intent_classifier import IntentClassifierAgent, IntentResult
from agents.model_router import ModelRouter
from agents.rag_agent import RAGAgent, RetrievalResult
from agents.response_generator import ResponseGeneratorAgent, GenerationResult
from config import ADAPTER_DIR, FAST_ADAPTER_DIR, FAST_MODEL_ID

TRUST_THRESHOLD = 0.6


@dataclass
class PipelineResult:
    query: str
    intent: IntentResult
    retrieval: RetrievalResult
    generation: GenerationResult
    total_pipeline_ms: float
    intent_ms: float = 0.0
    rag_ms: float = 0.0
    generation_ms: float = 0.0
    guardrail_ms: float = 0.0
    guardrail_input_flags: list = field(default_factory=list)
    guardrail_output_flags: list = field(default_factory=list)
    blocked: bool = False
    clarification: bool = False
    route: str = "expert"
    route_reason: str = "single-model deployment"
    trust_score: float = 1.0
    escalated: bool = False
    timestamp: float = field(default_factory=time.time)


def _early_result(query: str, response: str, label: str, t0: float,
                  flags: list, **kw) -> PipelineResult:
    now = time.perf_counter()
    return PipelineResult(
        query=query,
        intent=IntentResult(intent=label, confidence=1.0, matched_keywords=[]),
        retrieval=RetrievalResult(chunks=[], query=query, intent=label,
                                  retrieval_method="skipped"),
        generation=GenerationResult(response=response, tokens_generated=0,
                                    inference_time_ms=0.0, tokens_per_second=0.0,
                                    model_used=label),
        total_pipeline_ms=round((now - t0) * 1000, 2),
        guardrail_ms=round((now - t0) * 1000, 2),
        guardrail_input_flags=flags,
        **kw,
    )


class TelecomOrchestrator:
    def __init__(self, model_path: str = ADAPTER_DIR):
        print("[Orchestrator] Initializing agents...")
        self.guardrails = GuardrailsAgent()
        self.clarity = ClarityAgent()
        self.classifier = IntentClassifierAgent()
        self.rag = RAGAgent(use_chromadb=True)
        self.router = ModelRouter()
        self.generator = ResponseGeneratorAgent(model_path=model_path)

        # Fast lane activates only when a small-model adapter has been trained
        self.fast_generator = None
        if (Path(FAST_ADAPTER_DIR) / "adapter_config.json").exists():
            print(f"[Orchestrator] Fast lane: loading {FAST_MODEL_ID} + adapter...")
            self.fast_generator = ResponseGeneratorAgent(
                model_path=FAST_ADAPTER_DIR, base_model_id=FAST_MODEL_ID)
            print("[Orchestrator] Model router ACTIVE (fast + expert lanes).")
        else:
            print("[Orchestrator] Single-model mode (no fast adapter found).")
        print("[Orchestrator] Ready.")

    def process(self, query: str, history: str | None = None) -> PipelineResult:
        t0 = time.perf_counter()

        gin = self.guardrails.check_input(query)
        if gin.blocked:
            return _early_result(gin.masked_query, BLOCKED_RESPONSE,
                                 "guardrails (blocked before model)", t0,
                                 gin.flags, blocked=True)
        safe_query = gin.masked_query

        cl = self.clarity.check(safe_query, history)
        if not cl.clear:
            return _early_result(safe_query, cl.clarifying_question,
                                 f"clarity agent ({cl.reason})", t0,
                                 gin.flags, clarification=True)
        t_pre = time.perf_counter()

        routing_text = f"{history} {safe_query}" if history else safe_query
        intent_result = self.classifier.classify(routing_text)
        t1 = time.perf_counter()

        retrieval_result = self.rag.retrieve(routing_text, intent_result.intent, top_k=3)
        t2 = time.perf_counter()

        route, route_reason = "expert", "single-model deployment"
        gen_agent = self.generator
        if self.fast_generator is not None:
            decision = self.router.route(safe_query, intent_result.confidence)
            route, route_reason = decision.route, decision.reason
            if route == "fast":
                gen_agent = self.fast_generator

        gen_result = gen_agent.generate(
            safe_query, retrieval_result.chunks, intent_result.intent, history=history)
        t3 = time.perf_counter()

        gout = self.guardrails.check_output(gen_result.response, retrieval_result.chunks)
        gen_result.response = gout.response
        t4 = time.perf_counter()

        # Trust score — cheap signals we already have, gating human escalation
        trust = 1.0
        trust -= 0.35 * sum(1 for f in gout.flags if f.startswith("unverified_amount"))
        trust -= 0.15 * sum(1 for f in gout.flags if "pii_leak" in f)
        if intent_result.confidence < 0.5:
            trust -= 0.2
        if not retrieval_result.chunks:
            trust -= 0.1
        trust = max(0.0, round(trust, 2))

        return PipelineResult(
            query=safe_query,
            intent=intent_result,
            retrieval=retrieval_result,
            generation=gen_result,
            total_pipeline_ms=round((t4 - t0) * 1000, 1),
            intent_ms=round((t1 - t_pre) * 1000, 2),
            rag_ms=round((t2 - t1) * 1000, 1),
            generation_ms=round((t3 - t2) * 1000, 1),
            guardrail_ms=round(((t_pre - t0) + (t4 - t3)) * 1000, 2),
            guardrail_input_flags=gin.flags,
            guardrail_output_flags=gout.flags,
            route=route,
            route_reason=route_reason,
            trust_score=trust,
            escalated=trust < TRUST_THRESHOLD,
        )
