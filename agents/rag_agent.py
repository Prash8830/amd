"""RAG Retrieval Agent — retrieves telecom KB chunks via ChromaDB + sentence-transformers."""

from __future__ import annotations
from dataclasses import dataclass, field

TELECOM_KB = [
    {"id": "kb_billing_1", "text": "Bills are generated monthly. Payment is due 21 days after the billing cycle closes. Late fees of $5 apply after the due date.", "category": "billing"},
    {"id": "kb_billing_2", "text": "AutoPay discount of $5/month applies when you enroll bank account for automatic payment. Credit card autopay does not qualify for the discount.", "category": "billing"},
    {"id": "kb_billing_3", "text": "Billing disputes must be submitted within 60 days of the bill date. Disputes can be filed online or by calling billing support.", "category": "billing"},
    {"id": "kb_network_1", "text": "5G coverage is available in 300+ cities. Our 5G network uses both sub-6GHz and mmWave technology. Check coverage map for your area.", "category": "network"},
    {"id": "kb_network_2", "text": "Data throttling applies after high-speed data cap is reached. Speeds may be reduced to 1.5 Mbps for the rest of the billing cycle.", "category": "network"},
    {"id": "kb_network_3", "text": "VoLTE (Voice over LTE) provides HD voice quality and simultaneous voice and data. Requires VoLTE-capable device and plan.", "category": "network"},
    {"id": "kb_device_1", "text": "Device unlock requests are processed within 2 business days. Device must be fully paid off and account in good standing.", "category": "device"},
    {"id": "kb_device_2", "text": "Device Protection plan covers accidental damage, loss, and theft. Deductibles range from $29 to $249 depending on device tier.", "category": "device"},
    {"id": "kb_plan_1", "text": "Unlimited Premium plan includes 50GB premium data, 15GB hotspot, HD streaming, and international texting to 200+ countries.", "category": "plan"},
    {"id": "kb_plan_2", "text": "Multi-line discounts: 2nd line $10 off, 3rd and 4th lines $20 off each per month. Applies to qualifying unlimited plans.", "category": "plan"},
    {"id": "kb_account_1", "text": "Account PIN is required for number porting, account changes, and store visits. PIN can be changed in account security settings.", "category": "account"},
    {"id": "kb_account_2", "text": "Number porting takes 1 business day. Keep current service active until port is complete. Port confirmation will be sent via SMS.", "category": "account"},
]


@dataclass
class RetrievalResult:
    chunks: list[dict]
    query: str
    intent: str
    retrieval_method: str = "chromadb"


class RAGAgent:
    """RAG agent using ChromaDB for semantic retrieval with sentence-transformers embeddings."""

    def __init__(self, use_chromadb: bool = True):
        self._collection = None
        self._use_chromadb = use_chromadb
        self._embedder = None
        self._bm25 = BM25Index([item["text"] for item in TELECOM_KB])
        self._id_to_idx = {item["id"]: i for i, item in enumerate(TELECOM_KB)}
        if use_chromadb:
            self._init_chromadb()

    def _init_chromadb(self):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            client = chromadb.Client()
            self._collection = client.get_or_create_collection("telecom_kb")

            if self._collection.count() == 0:
                docs = [item["text"] for item in TELECOM_KB]
                ids = [item["id"] for item in TELECOM_KB]
                embeddings = self._embedder.encode(docs).tolist()
                metadatas = [{"category": item["category"]} for item in TELECOM_KB]
                self._collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)
        except Exception as e:
            print(f"[RAGAgent] ChromaDB init failed ({e}), falling back to keyword retrieval.")
            self._collection = None

    def retrieve(self, query: str, intent: str, top_k: int = 3) -> RetrievalResult:
        """Hybrid retrieval with query fusion.

        BM25 catches exact lexical matches (codes, hardware model numbers —
        where embeddings are weak); the vector index catches semantic
        similarity. Each query variant is ranked by both, and all rankings are
        fused with Reciprocal Rank Fusion. Degrades to BM25-only when the
        vector store is unavailable.
        """
        variants = self._query_variants(query, intent)
        rankings = []
        vector_used = False
        for v in variants:
            rankings.append(self._bm25.rank(v))
            if self._collection is not None and self._embedder is not None:
                vr = self._vector_rank(v)
                if vr:
                    rankings.append(vr)
                    vector_used = True

        fused = _rrf(rankings)
        chunks = [{"text": TELECOM_KB[i]["text"], "id": TELECOM_KB[i]["id"],
                   "score": round(score, 4)} for i, score in fused[:top_k]]
        method = ("hybrid RRF (bm25+vector, query fusion)" if vector_used
                  else "bm25 (query fusion)")
        return RetrievalResult(chunks=chunks, query=query, intent=intent,
                               retrieval_method=method)

    def _query_variants(self, query: str, intent: str) -> list[str]:
        """Query fusion: the raw query plus an intent-expanded variant."""
        variants = [query]
        from agents.intent_classifier import INTENTS
        kws = INTENTS.get(intent, [])
        if kws:
            variants.append(f"{intent} {' '.join(kws[:6])} {query}")
        return variants

    def _vector_rank(self, query: str) -> list[int]:
        """Document indices ordered by vector similarity."""
        try:
            emb = self._embedder.encode([query]).tolist()
            results = self._collection.query(query_embeddings=emb,
                                             n_results=len(TELECOM_KB))
            return [self._id_to_idx[mid] for mid in results["ids"][0]
                    if mid in self._id_to_idx]
        except Exception:
            return []


def _rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple rankings of doc indices."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Index:
    """Minimal Okapi BM25 over the KB — pure Python, no dependencies.
    Exact lexical matching is what embeddings miss on alphanumeric tokens
    like "B-204" or "HG-2410"."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        import math
        import re
        self._tok = lambda t: re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t.lower())
        self.k1, self.b = k1, b
        self.doc_tokens = [self._tok(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.N = len(docs)
        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1)
                    for t, n in df.items()}

    def rank(self, query: str) -> list[int]:
        q_terms = self._tok(query)
        scores = []
        for i, toks in enumerate(self.doc_tokens):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            scores.append((s, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [i for s, i in scores if s > 0] or [i for _, i in scores]
