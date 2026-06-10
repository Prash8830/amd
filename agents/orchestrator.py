"""Multi-agent orchestrator — coordinates intent → RAG → generation pipeline."""

from __future__ import annotations
import time
from dataclasses import dataclass, field

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
    timestamp: float = field(default_factory=time.time)


class TelecomOrchestrator:
    """Coordinates the three-agent pipeline for telecom support queries."""

    def __init__(self, model_path: str = ADAPTER_DIR):
        print("[Orchestrator] Initializing agents...")
        self.classifier = IntentClassifierAgent()
        self.rag = RAGAgent(use_chromadb=True)
        self.generator = ResponseGeneratorAgent(model_path=model_path)
        print("[Orchestrator] Ready.")

    def process(self, query: str) -> PipelineResult:
        t0 = time.perf_counter()
        intent_result = self.classifier.classify(query)
        t1 = time.perf_counter()
        retrieval_result = self.rag.retrieve(query, intent_result.intent, top_k=3)
        t2 = time.perf_counter()
        gen_result = self.generator.generate(query, retrieval_result.chunks, intent_result.intent)
        t3 = time.perf_counter()

        return PipelineResult(
            query=query,
            intent=intent_result,
            retrieval=retrieval_result,
            generation=gen_result,
            total_pipeline_ms=round((t3 - t0) * 1000, 1),
            intent_ms=round((t1 - t0) * 1000, 2),
            rag_ms=round((t2 - t1) * 1000, 1),
            generation_ms=round((t3 - t2) * 1000, 1),
        )
