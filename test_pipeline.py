"""Notebook-friendly pipeline test — runs canned queries, no stdin needed.

Run: python test_pipeline.py
"""

# Unsloth must be imported before transformers/trl so its patches apply
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

from agents.orchestrator import TelecomOrchestrator

TEST_QUERIES = [
    "Why is my bill higher than usual this month?",
    "I have no signal in my area, what should I do?",
    "How do I unlock my phone?",
    "Can I upgrade to the unlimited plan?",
]


def main():
    orchestrator = TelecomOrchestrator()

    for query in TEST_QUERIES:
        result = orchestrator.process(query)
        print("=" * 70)
        print(f"Query:    {query}")
        print(f"Intent:   {result.intent.intent} (confidence: {result.intent.confidence})")
        print(f"RAG:      {len(result.retrieval.chunks)} chunks via {result.retrieval.retrieval_method}")
        print(f"Model:    {result.generation.model_used}")
        print(f"Speed:    {result.generation.tokens_per_second} tok/s | {result.generation.inference_time_ms}ms inference | {result.total_pipeline_ms}ms total")
        print(f"Response: {result.generation.response}")
        print()

    print("All test queries completed.")


if __name__ == "__main__":
    main()
