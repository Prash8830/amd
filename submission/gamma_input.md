# Gamma.app input v2 — TruthLine deck (card-by-card)

## Gamma UI settings (prompts cannot control these — set manually)
- Theme: pick a DARK theme in the theme picker; set accent color to #ED1C24
- AI images: OFF (or minimal/abstract) — numbers and our diagram are the visuals
- Text density: medium · Format: presentation 16:9 · export PDF for submission
- Card 3: insert the TruthLine pipeline diagram image (from the chat widget
  "truthline_final_slide_flow_diagram") full width; beats below/beside it

## Content box (--- = card breaks)

TruthLine — the support model that cannot be bluffed

Track 3 · Fine-Tuning · FINETUNING_002 · AMD Instinct MI300X
Prashant Patil — research, architecture, fine-tuning, engineering
22% → 94% measured accuracy on proprietary telecom knowledge
60 seconds — full LoRA fine-tune on AMD
Built end-to-end this week: domain-expert models, agentic pipeline, MCP tool layer, vLLM serving path, live AMD telemetry
---
Every telco has a B-204. No public model knows what it means.

We asked base Qwen3-14B about internal billing code B-204. It answered instantly and confidently: "a prorated adjustment credit — you were overcharged and are being credited." INVENTED. That customer now expects a refund that doesn't exist — a complaint, an escalation, a churn risk, manufactured by the AI itself.
TruthLine's domain-expert model, same question: "B-204 is a prorated plan-change adjustment — a one-time charge after a mid-cycle plan switch." CORRECT — from the weights.
The trap: generic LLMs never say "I don't know" about internal codes, router hardware, or error codes — and no prompt can fix what was never in the pretraining data.
Who needs this: telecom contact centers (agent-assist + self-service) — and any enterprise whose vocabulary isn't on the public internet.
---
Six decisions that kill hallucination

[PLACE THE TRUTHLINE PIPELINE DIAGRAM IMAGE HERE — full width]
1. Ask, don't guess — guardrails mask PII and block injections; a clarity gate turns vague queries into clarifying questions. Most hallucinations die here, at zero GPU cost.
2. Domain truth in the weights — a 60-second LoRA fine-tune creates the domain-expert model.
3. Right-sized compute — a router sends simple FAQs to the fine-tuned 1.5B, proprietary codes to the 14B expert lane (served in-process or via vLLM as an API endpoint). Never a bulldozer for a thumbtack.
4. Facts in the knowledge fabric — hybrid retrieval (BM25 + vector, RRF query fusion) plus live enterprise data over our MCP server (outage feed).
5. No answer ships untrusted — output guardrails flag unverified claims; a trust gate routes low-trust answers via MCP to the right on-call domain expert, never to the customer.
6. The data flywheel — every thumbs-up becomes ground truth: served instantly from tier-zero semantic cache (zero GPU, 10 ms) and auto-merged into the next 60-second retrain. Every approval becomes tomorrow's weights.
---
Proof, not promises

Held-out accuracy, 18 paraphrased questions: proprietary billing codes 0% → 80% · error codes 0% → 100% · hardware 0% → 100% · public telecom 80% → 100% · TOTAL 22% → 94%
The experimental design: the proprietary facts are synthetic — we invented them — so the base model cannot know them. The improvement is provable, not anecdotal.
60 seconds — full LoRA retrain on one MI300X (r=32, bf16, ~0.9% of parameters)
−74% tokens per answer (302 → 78) · −51% end-to-end latency
Entire stack — 14B expert + 1.5B fast lane + comparison copy — in under 35% of one 192 GB MI300X
98% GPU utilization at ~740 W during training — rocm-smi telemetry is built into the product, not bolted on
---
Built this week. Ready for Monday morning.

SHIPPED (not roadmap): fine-tuned 14B + 1.5B domain-expert models · LangGraph agentic pipeline (guardrails, clarity gate, trust gate) · MCP enterprise server (expert escalation routing, ITSM ticketing, live outage feed) · LangChain hybrid RAG (BM25 + vector, RRF query fusion) · tier-zero semantic cache · data flywheel · vLLM serving path · 5-tab enterprise console with live AMD telemetry · held-out evaluation harness
The demo: watch the base model hallucinate live, then TruthLine answer from its weights · PII masked in the trace · a thumbs-up turning into an instant zero-GPU cache hit · the GPU lighting up for a 60-second retrain
Scaling from here: batched vLLM serving under production load · MCP connectors into live CRM and billing systems · GRPO/DPO on rejection signals · corpus scaled with anonymized transcripts, every adapter eval-gated before promotion
TruthLine doesn't just answer correctly today — every interaction makes tomorrow's model better, for one minute of AMD GPU time a night.
