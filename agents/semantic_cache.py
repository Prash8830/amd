"""Semantic cache over the ground-truth DB — tier zero of the serving stack.

Every approved feedback pair is embedded; incoming queries are matched by
cosine similarity:

  sim >= hit_threshold     -> serve the human-approved answer directly.
                              Zero GPU, ~10 ms, trust 1.0 (a human validated it).
  assist <= sim < hit      -> not safe to serve verbatim; the approved pair is
                              injected as an extra grounding chunk so the model
                              adapts it (cache as evidence, not as answer).
  sim < assist_threshold   -> normal pipeline.

The flywheel improves this tier too: every thumbs-up grows the cache.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CacheResult:
    hit: bool = False
    assist: bool = False
    similarity: float = 0.0
    question: str = ""
    answer: str = ""


class SemanticCache:
    def __init__(self, embedder=None, hit_threshold: float = 0.90,
                 assist_threshold: float = 0.75):
        self.hit_threshold = hit_threshold
        self.assist_threshold = assist_threshold
        self._embedder = embedder  # reuse the RAG agent's sentence-transformer
        self._questions: list[str] = []
        self._answers: list[str] = []
        self._emb = None
        self.refresh()

    def size(self) -> int:
        return len(self._questions)

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def refresh(self) -> None:
        """Re-embed the approved set. Called at startup and on every thumbs-up."""
        from data.feedback_store import load_approved
        rows = load_approved()
        self._questions = [r["question"] for r in rows]
        self._answers = [r["answer"] for r in rows]
        if not self._questions:
            self._emb = None
            return
        self._ensure_embedder()
        import numpy as np
        self._emb = np.asarray(
            self._embedder.encode(self._questions, normalize_embeddings=True))

    def lookup(self, query: str) -> CacheResult:
        if self._emb is None or not query.strip():
            return CacheResult()
        q = self._embedder.encode([query], normalize_embeddings=True)[0]
        sims = self._emb @ q
        i = int(sims.argmax())
        sim = float(sims[i])
        if sim >= self.hit_threshold:
            return CacheResult(hit=True, similarity=round(sim, 3),
                               question=self._questions[i], answer=self._answers[i])
        if sim >= self.assist_threshold:
            return CacheResult(assist=True, similarity=round(sim, 3),
                               question=self._questions[i], answer=self._answers[i])
        return CacheResult(similarity=round(sim, 3))
