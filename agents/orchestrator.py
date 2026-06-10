"""Multi-agent orchestrator — coordinates intent → RAG → generation pipeline."""

from __future__ import annotations
import time
from dataclasses import dataclass, field

from agents.intent_classifier import IntentClassifierAgent, IntentResult
from agents.rag_agent import RAGAgent, RetrievalResult
from agents.response_generator import ResponseGeneratorAgent, GenerationResult


@dataclass
class PipelineResult:
    query: str
    intent: IntentResult
    retrieval: RetrievalResult
    generation: GenerationResult
    total_pipeline_ms: float
    timestamp: float = field(default_factory=time.time)


class TelecomOrchestrator:
    """Coordinates the three-agent pipeline for telecom support queries."""

    def __init__(self, model_path: str = "./models/qwen3-14b-telecom-qlora"):
        print("[Orchestrator] Initializing agents...")
        self.classifier = IntentClassifierAgent()
        self.rag = RAGAgent(use_chromadb=True)
        self.generator = ResponseGeneratorAgent(model_path=model_path)
        print("[Orchestrator] Ready.")

    def process(self, query: str) -> PipelineResult:
        t0 = time.perf_counter()

        intent_result = self.classifier.classify(query)
        retrieval_result = self.rag.retrieve(query, intent_result.intent, top_k=3)
        gen_result = self.generator.generate(query, retrieval_result.chunks, intent_result.intent)

        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        return PipelineResult(
            query=query,
            intent=intent_result,
            retrieval=retrieval_result,
            generation=gen_result,
            total_pipeline_ms=total_ms,
        )
