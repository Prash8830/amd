"""
Telecom Support Chatbot — AMD Hackathon
Entry point: python main.py [--mode api|cli|finetune]
"""

import argparse
import os
import sys

# ROCm allocator tuning (no GFX override — MI300X is natively supported)
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512")


def run_cli():
    from agents.orchestrator import TelecomOrchestrator

    print("=" * 60)
    print("  Telecom Support Chatbot — AMD ROCm Edition")
    print("  Powered by Qwen3-14B QLoRA + Multi-Agent Pipeline")
    print("=" * 60)

    orchestrator = TelecomOrchestrator()

    print("\nType your question (or 'quit' to exit):\n")
    while True:
        try:
            query = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        result = orchestrator.process(query)

        print(f"\n[Intent: {result.intent.intent} | confidence: {result.intent.confidence}]")
        print(f"[Retrieved {len(result.retrieval.chunks)} KB chunks via {result.retrieval.retrieval_method}]")
        print(f"[Model: {result.generation.model_used} | {result.generation.tokens_per_second} tok/s | {result.generation.inference_time_ms}ms]")
        print(f"\nAgent: {result.generation.response}\n")
        print(f"Pipeline latency: {result.total_pipeline_ms}ms")
        print("-" * 60)


def run_api():
    import uvicorn
    from api import create_app

    app = create_app()
    print("[API] Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_finetune():
    import finetune
    finetune.train()


def run_streamlit():
    import subprocess
    # CORS/XSRF off so the app works behind jupyter-server-proxy
    # (reach it at <jupyter-base-url>/proxy/8501/)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom Support Chatbot")
    parser.add_argument("--mode", choices=["cli", "api", "finetune", "ui"], default="cli",
                        help="cli=interactive, api=FastAPI server, finetune=train model, ui=Streamlit")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "api":
        run_api()
    elif args.mode == "finetune":
        run_finetune()
    elif args.mode == "ui":
        run_streamlit()
