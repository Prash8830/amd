"""Central config — change BASE_MODEL_ID here (or via env var) to swap models.

Pipeline validation: Qwen/Qwen2.5-1.5B  (fast, small)
Final hackathon run: Qwen/Qwen3-14B     (full size, current default)
"""

import os

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen3-14B")

# Adapter path is model-specific — a 1.5B adapter can't load onto a 14B base
_model_slug = BASE_MODEL_ID.split("/")[-1].lower()
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", f"./models/telecom-qlora-{_model_slug}")

# 4-bit saves VRAM but adds dequant overhead — pointless on MI300X (206GB) for a
# small model. Set true when running Qwen3-14B if VRAM is ever a concern.
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"

MAX_SEQ_LENGTH = 1024  # our telecom samples are short; smaller = faster
