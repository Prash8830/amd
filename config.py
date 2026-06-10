"""Central config — change BASE_MODEL_ID here (or via env var) to swap models.

Pipeline validation: Qwen/Qwen2.5-1.5B  (fast, small)
Final hackathon run: Qwen/Qwen3-14B     (full size)
"""

import os

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B")

# Fine-tuned adapter output/load path (model-agnostic name so swaps don't break it)
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./models/telecom-qlora")

MAX_SEQ_LENGTH = 2048
