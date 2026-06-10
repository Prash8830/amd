"""Multi-agent orchestrator — guardrails → intent → RAG → generation → guardrails."""

from __future__ import annotations
import time
from dataclasses import dataclass, field

from agents.guardrails import GuardrailsAgent, BLOCKED_RESPONSE
from agents.intent_classifier import IntentClassifierAgent, IntentResult
from agents.rag_agent import RAGAgent, RetrievalResult
from agents.response_generator import ResponseGeneratorAgent, GenerationResult
from config import ADAPTER_DIR


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
    timestamp: float = field(default_factory=time.time)


class TelecomOrchestrator:
    """Coordinates the four-agent pipeline for telecom support queries.

    process() optionally takes recent conversation history so follow-up
    questions ("the light is red") resolve against earlier turns ("my router
    is an HG-2410"). History feeds intent classification, retrieval, and the
    generation prompt.
    """

    def __init__(self, model_path: str = ADAPTER_DIR):
        print("[Orchestrator] Initializing agents...")
        self.guardrails = GuardrailsAgent()
        self.classifier = IntentClassifierAgent()
        self.rag = RAGAgent(use_chromadb=True)
        self.generator = ResponseGeneratorAgent(model_path=model_path)
        print("[Orchestrator] Ready.")

    def process(self, query: str, history: str | None = None) -> PipelineResult:
        t0 = time.perf_counter()

        gin = self.guardrails.check_input(query)
        t_guard_in = time.perf_counter()

        if gin.blocked:
            return PipelineResult(
                query=gin.masked_query,
                intent=IntentResult(intent="blocked", confidence=1.0, matched_keywords=[]),
                retrieval=RetrievalResult(chunks=[], query=gin.masked_query,
                                          intent="blocked", retrieval_method="skipped"),
                generation=GenerationResult(
                    response=BLOCKED_RESPONSE, tokens_generated=0,
                    inference_time_ms=0.0, tokens_per_second=0.0,
                    model_used="guardrails (blocked before model)"),
                total_pipeline_ms=round((t_guard_in - t0) * 1000, 2),
                guardrail_ms=round((t_guard_in - t0) * 1000, 2),
                guardrail_input_flags=gin.flags,
                blocked=True,
            )

        safe_query = gin.masked_query
        # History improves intent + retrieval for elliptical follow-ups
        routing_text = f"{history} {safe_query}" if history else safe_query

        intent_result = self.classifier.classify(routing_text)
        t1 = time.perf_counter()
        retrieval_result = self.rag.retrieve(routing_text, intent_result.intent, top_k=3)
        t2 = time.perf_counter()
        gen_result = self.generator.generate(
            safe_query, retrieval_result.chunks, intent_result.intent, history=history)
        t3 = time.perf_counter()

        gout = self.guardrails.check_output(gen_result.response, retrieval_result.chunks)
        gen_result.response = gout.response
        t4 = time.perf_counter()

        return PipelineResult(
            query=safe_query,
            intent=intent_result,
            retrieval=retrieval_result,
            generation=gen_result,
            total_pipeline_ms=round((t4 - t0) * 1000, 1),
            intent_ms=round((t1 - t_guard_in) * 1000, 2),
            rag_ms=round((t2 - t1) * 1000, 1),
            generation_ms=round((t3 - t2) * 1000, 1),
            guardrail_ms=round(((t_guard_in - t0) + (t4 - t3)) * 1000, 2),
            guardrail_input_flags=gin.flags,
            guardrail_output_flags=gout.flags,
        )
