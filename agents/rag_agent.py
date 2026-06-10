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
        if self._collection is not None and self._embedder is not None:
            return self._chromadb_retrieve(query, intent, top_k)
        return self._keyword_retrieve(query, intent, top_k)

    def _chromadb_retrieve(self, query: str, intent: str, top_k: int) -> RetrievalResult:
        query_embedding = self._embedder.encode([query]).tolist()
        results = self._collection.query(query_embeddings=query_embedding, n_results=top_k)
        chunks = [
            {"text": doc, "id": mid, "score": 1 - dist}
            for doc, mid, dist in zip(
                results["documents"][0],
                results["ids"][0],
                results["distances"][0],
            )
        ]
        return RetrievalResult(chunks=chunks, query=query, intent=intent, retrieval_method="chromadb")

    def _keyword_retrieve(self, query: str, intent: str, top_k: int) -> RetrievalResult:
        query_lower = query.lower()
        scored = []
        for item in TELECOM_KB:
            score = sum(word in item["text"].lower() for word in query_lower.split())
            if item["category"] == intent:
                score += 2
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [{"text": item["text"], "id": item["id"], "score": score} for score, item in scored[:top_k]]
        return RetrievalResult(chunks=chunks, query=query, intent=intent, retrieval_method="keyword")
