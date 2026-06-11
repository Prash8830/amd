# TruthLine — Demo Video Script (~4 minutes)

**Recording setup:** screen-record the browser (TruthLine UI) + the Jupyter
terminal running the MCP server visible when called for. Calm pace; let the
UI breathe for 1–2 seconds after each result so viewers can read.

**Pre-recording checklist (one GPU session):**
- [ ] MCP server running in a visible terminal (`python mcp_server/telecom_mcp.py`)
- [ ] Streamlit app freshly started AFTER the MCP server (sidebar shows MCP connected)
- [ ] Both adapters present (14B + 1.5B) so the router is active
- [ ] `eval_results.json` present (Model-quality tab populated)
- [ ] Feedback file cleared for a clean flywheel demo: `rm -f feedback/ground_truth.jsonl`, restart app
- [ ] Second notebook cell ready with `!rocm-smi` for the proof moment

---

### 0:00–0:25 — The problem (Support console)

> "Generic LLMs hallucinate on proprietary telecom knowledge. Let me show you,
> live, on AMD hardware."

- Sidebar: flip ON "hallucination check".
- Ask: **`What does billing code B-204 mean on my invoice?`**
- Open the base-model expander first:
> "The base Qwen3-14B — running locally on our MI300X — invents a refund
> credit. Confident, plausible, wrong. A real customer now expects money back."
- Point at the main answer:
> "The same model after our fine-tune: B-204 is a prorated plan-change
> adjustment — correct, from the weights. That fine-tune took sixty seconds
> on the MI300X."

### 0:25–1:10 — The pipeline (trace open)

- Open the **pipeline trace** of the B-204 answer. Read it bottom-up briefly:
> "Every answer carries its audit trail: guardrails, intent, retrieval, the
> model-router decision — this one went to the 14B expert lane — and a trust
> score."
- Ask: **`My number is 555-867-5309 and my internet is very slow`**
- Trace: point at `pii_masked:phone` and `MCP tools: get_outage_status`:
> "PII is masked before the model or logs ever see it. And notice — the agent
> called our enterprise MCP server for the live network status."
- **Cut to the terminal**: show the MCP server logging `CallToolRequest`:
> "That's a real Model Context Protocol server — agents calling enterprise
> tools, not just an LLM answering."

### 1:10–1:40 — Clarity gate + model router

- Ask: **`it's not working`**
> "Vague query? TruthLine asks instead of guessing — most hallucinations die
> right here, at zero GPU cost."
- Ask: **`When is my payment due?`** — open trace, point at `route: fast`:
> "Simple FAQ — the router sends it to our fine-tuned 1.5B. Don't use a
> bulldozer for a thumbtack. Proprietary codes go to the 14B expert. Both
> live in a third of one MI300X."

### 1:40–2:10 — The flywheel + semantic cache

- Click **👍** on the B-204 answer.
- Ask: **`What is the B-204 code on my bill?`** (reworded!)
- Trace shows `route: cache — ground-truth hit`:
> "I approved that answer once. Asked again — answered from the human-approved
> ground-truth store in ten milliseconds, zero GPU. And at the next nightly
> retrain — one minute on this card — that approval becomes model weights.
> User feedback literally becomes the model."

### 2:10–2:40 — Escalation + governance (Governance tab)

- Show the human review queue (have one escalated item from earlier, or ask a
  query that triggers `unverified_amount`):
> "Low-trust answers never reach the customer — they land here, and MCP looks
> up the right on-call domain expert: network issue, network engineer."
- Show the guardrail activity table:
> "Full audit trail: every masked number, every flagged claim."

### 2:40–3:20 — The evidence (Model quality tab)

- Show the accuracy chart:
> "Our eval: eighteen held-out, paraphrased questions. The internal codes are
> synthetic — we invented them — so the base model *cannot* know them. That's
> what makes this measurable: twenty-two percent base, **ninety-four percent**
> fine-tuned. Tokens per answer down seventy-four percent."

### 3:20–3:50 — AMD proof (Observability tab + notebook)

- Show live telemetry; then run `!rocm-smi` in the notebook:
> "Everything you saw runs on one AMD Instinct MI300X — no external APIs.
> Live rocm-smi telemetry is built into the product. During training this
> card hit ninety-eight percent utilization at seven hundred forty watts."
- (Best version: kick off `python main.py --mode finetune` and show the GPU
  charts spike + loss printing — "this is a full retrain, watch it finish.")

### 3:50–4:10 — Close

> "TruthLine: fine-tuned domain truth in the weights, facts in the database,
> tools on the protocol, humans in the loop — and a flywheel that makes
> tomorrow's model better for one minute of GPU time a night. Built end-to-end
> this week, on AMD."

---

**Backup plan if GPU/UI unavailable at recording time:** narrate over the
screenshot set in the same order; the terminal MCP log and the eval table
carry the proof.
