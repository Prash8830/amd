"""Notebook-friendly pipeline test — runs canned queries, no stdin needed.

Run: python test_pipeline.py
"""

from utils.steps import StepTracker

TEST_QUERIES = [
    "Why is my bill higher than usual this month?",
    "I have no signal in my area, what should I do?",
    "How do I unlock my phone?",
    "Can I upgrade to the unlimited plan?",
]


def main():
    steps = StepTracker(total=2 + len(TEST_QUERIES), title="PIPELINE TEST  ·  intent → RAG → generation")

    with steps.step("Import agents (torch/transformers load here)"):
        from agents.orchestrator import TelecomOrchestrator

    with steps.step("Initialize orchestrator (model loads to GPU here)"):
        orchestrator = TelecomOrchestrator()

    for i, query in enumerate(TEST_QUERIES, 1):
        with steps.step(f'Query {i}: "{query}"') as s:
            result = orchestrator.process(query)
            s.note(f"intent:   {result.intent.intent} (confidence {result.intent.confidence})")
            s.note(f"RAG:      {len(result.retrieval.chunks)} chunks via {result.retrieval.retrieval_method}")
            s.note(f"model:    {result.generation.model_used}")
            s.note(f"speed:    {result.generation.tokens_per_second} tok/s · {result.generation.inference_time_ms}ms infer · {result.total_pipeline_ms}ms total")
            s.note(f"response: {result.generation.response}")

    steps.done()


if __name__ == "__main__":
    main()
