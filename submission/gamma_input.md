# Gamma.app input — TruthLine deck (card-by-card format)

## Part 1 — paste into Gamma's prompt/instructions field

Enterprise SaaS product deck for a technical AI jury (TCS × AMD hackathon).
Dark theme, white text, AMD red (#ED1C24) accent used sparingly for key
numbers and arrows. Confident product-launch tone, not academic. Big headline
numbers as visual anchors. Flat minimal diagrams, no stock photos. Use ONLY
the numbers given — do not invent content. Keep this exact terminology:
knowledge fabric (never "knowledge base"), domain-expert model, curated domain
corpus, human-in-the-loop signal, data flywheel, trust gate, tier-zero
serving, right-sized compute, agentic pipeline.

## Part 2 — paste into the content box (--- = card breaks)

TruthLine — Telco-specific Customer LLM

Track 3 (Fine-Tuning) · Use case FINETUNING_002 · AMD Instinct MI300X
Team <NAME> — Prashant Patil (research, architecture, fine-tuning, engineering)
22% → 94% measured accuracy on proprietary telecom knowledge — with 60 seconds of fine-tuning on AMD
All code built during the hackathon · open-source stack · concepts adapted from the author's patent-pending TruthGate design
---
The problem: confident hallucination

Asked about internal billing code B-204, base Qwen3-14B answered: "a prorated adjustment credit — you were overcharged and are being credited" — INVENTED. The customer now expects a refund that doesn't exist.
TruthLine's domain-expert model: "B-204 is a prorated plan-change adjustment — a one-time charge after a mid-cycle plan switch" — CORRECT
Generic LLMs never say "I don't know" about internal codes, router hardware, or error codes — they invent plausible answers
No prompt can fix it: proprietary knowledge was never in the pretraining data
Target users: telecom contact centers (agent-assist + self-service); any enterprise with a proprietary vocabulary
---
The architecture, told as a story

Stage 1 — Where everyone starts: customer query → LLM → answer. This is where hallucination lives.
Stage 2 — Domain truth in the weights: a 60-second LoRA fine-tune turns the LLM into a domain-expert model (1.5B)
Stage 3 — Right-sized compute: a model router sends simple queries to the 1.5B fast lane, proprietary codes and troubleshooting to the fine-tuned 14B expert lane — never a bulldozer for a thumbtack
Stage 4 — Facts in the knowledge fabric: RAG grounding + live enterprise data over MCP (outage feed, internal systems)
Stage 5 — Duty of care: guardrail layer (PII masking, injection block) and clarity gate (ask, don't guess) on the way in; trust gate on the way out — low-trust answers route via MCP to the on-call domain expert, never to the customer
Stage 6 — The data flywheel: every thumbs-up becomes ground truth → serves repeat questions from tier-zero semantic cache (zero GPU, 10 ms) AND auto-merges into the next 60-second LoRA retrain. Every approval becomes tomorrow's weights.
---
Model insights & AMD efficiency

Held-out accuracy (18 paraphrased questions): proprietary billing codes 0% → 80% · error codes 0% → 100% · hardware 0% → 100% · public telecom 80% → 100% · TOTAL 22% → 94%
The proprietary facts are synthetic — invented by us — so the base model cannot know them: the improvement is provable, not anecdotal
60 seconds: full LoRA retrain on one MI300X (r=32, bf16, ~0.9% of parameters trained)
−74% tokens per answer (302 → 78) · −51% end-to-end latency
Entire serving stack — 14B + 1.5B + comparison copy — resident in under 35% of one 192 GB MI300X
98% GPU utilization at ~740 W during training, observed live via rocm-smi telemetry built into the product
---
Impact, demo, and the road ahead

Business impact: trusted deflection via trust gate + MCP expert escalation · ~4× serving throughput per GPU from token efficiency · repeat questions converge to zero-GPU tier-zero serving · the model improves nightly from human-in-the-loop signal, no ML team required
Demo: live base-vs-domain-expert hallucination comparison · pipeline traces showing PII masking and MCP tool calls · thumbs-up → instant tier-zero cache hit · live rocm-smi telemetry during a 60-second retrain
Road ahead: vLLM/SGLang batched serving on ROCm · MCP connectors into production CRM and billing · GRPO/DPO on rejection signals · LangGraph orchestration · scale the curated domain corpus with anonymized transcripts, eval-gated
TruthLine doesn't just answer correctly today — every interaction makes tomorrow's model better, for one minute of AMD GPU time a night.

## Notes
- Card 3 (architecture): after generation, replace Gamma's auto-diagram with
  our TruthLine flow diagram image if Gamma's rendering is weak. The staged
  Stage 1→6 text is written so it also reads well as a timeline layout.
- Submission rule: final PDF must be 3–5 slides. This input produces exactly 5.
- Fill in <NAME> before generating.
