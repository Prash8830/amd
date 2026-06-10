# Telecom Support Chatbot — AMD Hackathon

Fine-tuned **Qwen QLoRA** telecom customer support chatbot with multi-agent RAG pipeline and AMD GPU observability dashboard.

**Model selection** is centralized in `config.py` — currently `Qwen/Qwen2.5-1.5B` for fast pipeline validation. Swap to `Qwen/Qwen3-14B` for the final run by editing `config.py` or setting the env var:

```bash
BASE_MODEL_ID=Qwen/Qwen3-14B python main.py --mode finetune
```

## Architecture

```
User Query
    │
    ▼
Intent Classifier Agent  (keyword + regex, 5 intents)
    │
    ▼
RAG Retrieval Agent      (ChromaDB + sentence-transformers)
    │
    ▼
Response Generator Agent (Qwen3-14B QLoRA, 4-bit, AMD ROCm)
    │
    ▼
Streamlit Dashboard      (GPU util, VRAM, tok/s, latency)
```

## Quick Start (AMD ROCm Jupyter Notebook)

```bash
# 1. Clone
git clone https://github.com/Prash8830/amd.git
cd amd

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Fine-tune (fast with small model; minutes on MI300)
python main.py --mode finetune

# 3b. Test pipeline (notebook-friendly, no stdin needed)
python test_pipeline.py

# 3c. Measure accuracy: base vs fine-tuned on held-out questions
#     (internal billing codes / hardware / error codes — facts a base
#      model cannot know — plus public telecom questions)
python evaluate.py        # prints accuracy table, writes eval_results.json

# 3b. Interactive CLI
python main.py --mode cli

# 3c. FastAPI backend
python main.py --mode api

# 3d. Streamlit dashboard
python main.py --mode ui
# or: streamlit run app.py --server.port 8501
```

## From Jupyter Notebook

```python
!git clone https://github.com/Prash8830/amd.git
%cd amd
!pip install -r requirements.txt

# Fine-tune first
!python main.py --mode finetune

# Then run the UI
!python main.py --mode ui
```

## ROCm Notes

- Set `HSA_OVERRIDE_GFX_VERSION=11.0.0` for MI300/RX7900 series (already in code)
- Set `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512` to avoid OOM
- 4-bit QLoRA requires ~8GB VRAM minimum; 24GB recommended for Qwen3-14B

## Files

| File | Purpose |
|------|---------|
| `finetune.py` | QLoRA fine-tuning script |
| `main.py` | Entry point (cli/api/finetune/ui modes) |
| `app.py` | Streamlit observability dashboard |
| `api.py` | FastAPI backend |
| `agents/orchestrator.py` | Multi-agent pipeline coordinator |
| `agents/intent_classifier.py` | Intent classification agent |
| `agents/rag_agent.py` | ChromaDB RAG retrieval agent |
| `agents/response_generator.py` | Qwen3-14B generation agent |
| `data/telecom_dataset.py` | Synthetic telecom QA dataset |
| `utils/amd_metrics.py` | AMD ROCm GPU metrics via rocm-smi |
