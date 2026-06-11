"""FastAPI backend — exposes /chat and /health endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from agents.orchestrator import TelecomOrchestrator
from utils.amd_metrics import get_system_metrics

_orchestrator: TelecomOrchestrator | None = None


def get_orchestrator() -> TelecomOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TelecomOrchestrator()
    return _orchestrator


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    intent: str
    intent_confidence: float
    retrieval_method: str
    tokens_generated: int
    inference_time_ms: float
    tokens_per_second: float
    total_pipeline_ms: float
    model_used: str


def create_app() -> FastAPI:
    app = FastAPI(title="TruthLine API", version="1.0.0")

    @app.on_event("startup")
    async def startup():
        get_orchestrator()

    @app.get("/health")
    async def health():
        metrics = get_system_metrics()
        gpu = metrics["gpu"]
        return {
            "status": "ok",
            "cpu_pct": metrics["cpu_pct"],
            "ram_used_gb": metrics["ram_used_gb"],
            "gpu_available": gpu.available,
            "gpu_util_pct": gpu.gpu_utilization_pct,
            "vram_used_mb": gpu.vram_used_mb,
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        result = get_orchestrator().process(req.query)
        return ChatResponse(
            response=result.generation.response,
            intent=result.intent.intent,
            intent_confidence=result.intent.confidence,
            retrieval_method=result.retrieval.retrieval_method,
            tokens_generated=result.generation.tokens_generated,
            inference_time_ms=result.generation.inference_time_ms,
            tokens_per_second=result.generation.tokens_per_second,
            total_pipeline_ms=result.total_pipeline_ms,
            model_used=result.generation.model_used,
        )

    return app
