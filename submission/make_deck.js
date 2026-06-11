/* TruthLine deck generator — dark enterprise theme, animated architecture build.
   Usage: node make_deck.js animated|submission <outfile> */

const PptxGenJS = require("pptxgenjs");

const MODE = process.argv[2] || "animated";
const OUT = process.argv[3] || "TruthLine_deck.pptx";

// ── Palette ──────────────────────────────────────────────────────────────────
const BG = "121212";
const CARD = "1D1D1D";
const CARD_EDGE = "3A3A3A";
const TEXT = "F4F1EA";
const MUTED = "A9A399";
const RED = "ED1C24";
const RED_DARK = "7A1014";
const GREEN = "2FBF71";
const DIM_EDGE = "4A4A4A";
const DIM_TEXT = "8F8A82";

const HEAD = "Arial Black";
const BODY = "Calibri";

const pptx = new PptxGenJS();
pptx.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pptx.layout = "WIDE";

function newSlide() {
  const s = pptx.addSlide();
  s.background = { color: BG };
  return s;
}

function kicker(s, text, x = 0.55, y = 0.42, w = 11.0) {
  s.addShape(pptx.shapes.RECTANGLE, { x, y: y + 0.045, w: 0.14, h: 0.14, fill: { color: RED }, line: { type: "none" } });
  s.addText(text, { x: x + 0.24, y: y - 0.06, w, h: 0.34, fontFace: BODY, fontSize: 11.5, color: MUTED, charSpacing: 3, bold: true });
}

function card(s, x, y, w, h, opts = {}) {
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.07,
    fill: { color: opts.fill || CARD },
    line: { color: opts.edge || CARD_EDGE, width: opts.edgeW || 1 },
  });
}

// ════════════════════════════════ SLIDE 1 — TITLE ════════════════════════════
function slideTitle() {
  const s = newSlide();
  kicker(s, "TRACK 3 · FINE-TUNING · FINETUNING_002 · AMD INSTINCT MI300X");
  s.addText("TruthLine", { x: 0.5, y: 0.95, w: 12.3, h: 1.25, fontFace: HEAD, fontSize: 66, color: TEXT, bold: true });
  s.addText("The support model that cannot be bluffed.", { x: 0.55, y: 2.18, w: 12.0, h: 0.5, fontFace: BODY, fontSize: 22, color: RED, italic: true });

  // Stat callouts
  const stats = [
    ["22% → 94%", "measured accuracy on proprietary telecom knowledge"],
    ["60 s", "full LoRA fine-tune on one MI300X"],
    ["−74%", "tokens per answer vs base model"],
  ];
  const sw = 3.9, gap = 0.25;
  stats.forEach(([num, label], i) => {
    const x = 0.55 + i * (sw + gap);
    card(s, x, 3.05, sw, 1.85);
    s.addText(num, { x: x + 0.25, y: 3.3, w: sw - 0.5, h: 0.85, fontFace: HEAD, fontSize: i === 0 ? 33 : 40, color: TEXT, bold: true });
    s.addText(label, { x: x + 0.25, y: 4.2, w: sw - 0.5, h: 0.55, fontFace: BODY, fontSize: 12.5, color: MUTED });
  });

  s.addText([
    { text: "Built end-to-end this week:  ", options: { color: RED, bold: true } },
    { text: "fine-tuned domain-expert models · 7-stage agentic pipeline · MCP tool layer · hybrid RAG · semantic cache · data flywheel · vLLM serving path · live AMD telemetry", options: { color: TEXT } },
  ], { x: 0.55, y: 5.3, w: 12.2, h: 0.7, fontFace: BODY, fontSize: 14 });

  s.addText("Prashant Patil — research · architecture · fine-tuning · engineering        All code written during the hackathon · open-source stack · concepts adapted from the author's patent-pending TruthGate design",
    { x: 0.55, y: 6.75, w: 12.2, h: 0.4, fontFace: BODY, fontSize: 10.5, color: MUTED });
}

// ═══════════════════════════════ SLIDE 2 — PROBLEM ═══════════════════════════
function slideProblem() {
  const s = newSlide();
  kicker(s, "THE PROBLEM");
  s.addText("Every telco has a B-204. No public model knows what it means.",
    { x: 0.5, y: 0.78, w: 12.3, h: 0.95, fontFace: HEAD, fontSize: 27, color: TEXT, bold: true });

  // Left card — base model (wrong)
  card(s, 0.55, 1.95, 6.0, 2.9, { edge: RED, edgeW: 1.5 });
  s.addText("BASE QWEN3-14B", { x: 0.85, y: 2.15, w: 5.4, h: 0.3, fontFace: BODY, fontSize: 12, color: RED, bold: true, charSpacing: 2 });
  s.addText("“a prorated adjustment credit — you were overcharged and are being credited”",
    { x: 0.85, y: 2.55, w: 5.4, h: 1.0, fontFace: BODY, fontSize: 16, color: TEXT, italic: true });
  s.addText([
    { text: "INVENTED. ", options: { color: RED, bold: true } },
    { text: "The customer now expects a refund that doesn't exist — a complaint, an escalation, a churn risk, manufactured by the AI itself.", options: { color: MUTED } },
  ], { x: 0.85, y: 3.7, w: 5.4, h: 1.0, fontFace: BODY, fontSize: 13.5 });

  // Right card — TruthLine (correct)
  card(s, 6.8, 1.95, 6.0, 2.9, { edge: GREEN, edgeW: 1.5 });
  s.addText("TRUTHLINE DOMAIN-EXPERT MODEL", { x: 7.1, y: 2.15, w: 5.4, h: 0.3, fontFace: BODY, fontSize: 12, color: GREEN, bold: true, charSpacing: 2 });
  s.addText("“B-204 is a prorated plan-change adjustment — a one-time charge after a mid-cycle plan switch”",
    { x: 7.1, y: 2.55, w: 5.4, h: 1.0, fontFace: BODY, fontSize: 16, color: TEXT, italic: true });
  s.addText([
    { text: "CORRECT — from the weights. ", options: { color: GREEN, bold: true } },
    { text: "Same question, same GPU, sixty seconds of fine-tuning apart.", options: { color: MUTED } },
  ], { x: 7.1, y: 3.7, w: 5.4, h: 1.0, fontFace: BODY, fontSize: 13.5 });

  // Bottom trio
  const trio = [
    ["They never say “I don't know”", "Generic LLMs invent plausible answers for internal codes, router hardware, and error codes."],
    ["No prompt can fix it", "Proprietary knowledge was never in the pretraining data — context tricks don't create knowledge."],
    ["Who needs this", "Telecom contact centers (agent-assist + self-service) — and any enterprise whose vocabulary isn't on the public internet."],
  ];
  trio.forEach(([t, b], i) => {
    const x = 0.55 + i * 4.18;
    card(s, x, 5.15, 3.95, 1.8);
    s.addText(t, { x: x + 0.22, y: 5.32, w: 3.5, h: 0.55, fontFace: BODY, fontSize: 14, color: TEXT, bold: true });
    s.addText(b, { x: x + 0.22, y: 5.85, w: 3.55, h: 1.0, fontFace: BODY, fontSize: 11.5, color: MUTED });
  });
}

// ════════════════════════ ARCHITECTURE BUILD (stages 1-6) ════════════════════
// Fixed geometry; each element appears at its stage; newest = red edge.
const ROW = 2.5, BH = 0.95, ROW_TOP = 1.62, ROW_BOT = 3.42;
const NODES = [
  { id: "cust", label: "Customer\nquery", x: 0.5, y: ROW, w: 1.32, stage: 1 },
  { id: "llm", label: "Generic LLM", x: 5.6, y: ROW, w: 1.9, stage: 1, only: 1 },
  { id: "ft", label: "Domain-expert\nmodel (1.5B)", x: 5.6, y: ROW, w: 1.9, stage: 2, only: 2 },
  { id: "guard", label: "Guardrails\nPII · injection", x: 2.07, y: ROW, w: 1.45, stage: 5 },
  { id: "clarity", label: "Clarity gate\nask, don't guess", x: 3.77, y: ROW, w: 1.5, stage: 5 },
  { id: "router", label: "Model router\nright-sized", x: 5.6, y: ROW, w: 1.45, stage: 3, from: 3 },
  { id: "fast", label: "Fast lane\n1.5B fine-tuned", x: 7.45, y: ROW_TOP, w: 1.8, stage: 3, from: 3 },
  { id: "expert", label: "Expert lane\n14B · vLLM", x: 7.45, y: ROW_BOT, w: 1.8, stage: 3, from: 3 },
  { id: "trust", label: "Trust gate\nscore < 0.6 ↗", x: 9.65, y: ROW, w: 1.45, stage: 5 },
  { id: "ans", label: "Answer", x: 11.45, y: ROW_TOP, w: 1.7, stage: 1 },
  { id: "human", label: "On-call expert\nvia MCP", x: 11.45, y: ROW_BOT, w: 1.7, stage: 5 },
  { id: "fabric", label: "Knowledge fabric — hybrid RAG (BM25 + vector, RRF) · MCP outage feed", x: 4.35, y: 4.85, w: 6.0, stage: 4 },
  { id: "gt", label: "Ground-truth DB\napproved pairs", x: 2.0, y: 5.95, w: 2.2, h: 0.8, stage: 6 },
  { id: "cache", label: "Tier-zero cache\n0 GPU · 10 ms", x: 5.0, y: 5.95, w: 2.1, h: 0.8, stage: 6 },
  { id: "retrain", label: "60 s LoRA retrain\non MI300X", x: 7.9, y: 5.95, w: 2.2, h: 0.8, stage: 6 },
];

const STAGE_META = [
  null,
  ["WHERE EVERYONE STARTS", "Customer query → LLM → answer. This is where hallucination lives — the model answers whether it knows or not."],
  ["DOMAIN TRUTH IN THE WEIGHTS", "A 60-second LoRA fine-tune turns the generic LLM into a domain-expert model. B-204 now lives in the weights."],
  ["RIGHT-SIZED COMPUTE", "A router sends simple FAQs to the 1.5B fast lane, proprietary codes to the 14B expert lane — in-process or served via vLLM. Never a bulldozer for a thumbtack."],
  ["FACTS IN THE KNOWLEDGE FABRIC", "Hybrid retrieval (BM25 + vector, RRF query fusion) grounds every answer; live enterprise data arrives over our MCP server."],
  ["DUTY OF CARE", "Guardrails mask PII and block injections; the clarity gate asks instead of guessing; the trust gate routes low-trust answers via MCP to the on-call domain expert — never to the customer."],
  ["THE DATA FLYWHEEL", "Every thumbs-up becomes ground truth: served instantly from the tier-zero cache (zero GPU) and auto-merged into the next 60-second retrain. Every approval becomes tomorrow's weights."],
];

function mid(n) { const h = n.h || BH; return { cx: n.x + n.w / 2, cy: n.y + h / 2, r: n.x + n.w, l: n.x, t: n.y, b: n.y + h }; }

function arrow(s, x1, y1, x2, y2, color, w = 1.5, head = true) {
  s.addShape(pptx.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1,
    line: head ? { color, width: w, endArrowType: "triangle" } : { color, width: w },
  });
}

function drawNode(s, n, stage) {
  const isNew = (n.stage === stage);
  const active = n.stage <= stage;
  if (!active) return;
  const edge = isNew ? RED : CARD_EDGE;
  const tcol = isNew ? TEXT : (stage > n.stage ? TEXT : TEXT);
  const nh = n.id === "fabric" ? 0.7 : (n.h || BH);
  card(s, n.x, n.y, n.w, nh, { edge, edgeW: isNew ? 2 : 1 });
  const [t1, t2] = n.label.split("\n");
  if (t2 !== undefined && n.id !== "fabric") {
    s.addText([
      { text: t1 + "\n", options: { bold: true, fontSize: 12.5, color: tcol } },
      { text: t2, options: { fontSize: 10, color: isNew ? "FFB3B6" : MUTED } },
    ], { x: n.x + 0.06, y: n.y + 0.06, w: n.w - 0.12, h: BH - 0.12, fontFace: BODY, align: "center", valign: "middle", lineSpacing: 13 });
  } else {
    s.addText(n.label, { x: n.x + 0.1, y: n.y + 0.05, w: n.w - 0.2, h: nh - 0.1, fontFace: BODY, fontSize: n.id === "fabric" ? 12 : 12.5, bold: true, color: tcol, align: "center", valign: "middle" });
  }
}

function slideStage(stage) {
  const s = newSlide();
  kicker(s, `ARCHITECTURE · STEP ${stage} OF 6`);
  s.addText("Six decisions that kill hallucination", { x: 0.5, y: 0.72, w: 12.3, h: 0.6, fontFace: HEAD, fontSize: 25, color: TEXT, bold: true });

  const N = Object.fromEntries(NODES.map(n => [n.id, (n.id === "ans" && stage <= 2) ? { ...n, y: ROW } : n]));
  const vis = id => N[id].stage <= stage && (!N[id].only || N[id].only === stage);
  const ac = (...ids) => ids.some(id => N[id].stage === stage) ? RED : DIM_EDGE;

  // nodes
  Object.values(N).forEach(n => { if (!n.only || n.only === stage) drawNode(s, n, stage); });

  // arrows by stage topology
  const m = id => mid(N[id]);
  if (stage <= 2) {
    const model = stage === 1 ? "llm" : "ft";
    arrow(s, m("cust").r, m("cust").cy, N[model].x, m(model).cy, ac("cust", model));
    arrow(s, m(model).r, m(model).cy, N["ans"].x, ROW + BH / 2, ac(model));
  } else {
    // main chain: cust -> [guard -> clarity ->] router
    if (stage >= 5) {
      arrow(s, m("cust").r, m("cust").cy, N["guard"].x, m("guard").cy, ac("guard"));
      arrow(s, m("guard").r, m("guard").cy, N["clarity"].x, m("clarity").cy, ac("guard", "clarity"));
      arrow(s, m("clarity").r, m("clarity").cy, N["router"].x, m("router").cy, ac("clarity"));
    } else {
      arrow(s, m("cust").r, m("cust").cy, N["router"].x, m("router").cy, DIM_EDGE);
    }
    // router forks
    arrow(s, m("router").r, m("router").cy - 0.12, N["fast"].x, m("fast").cy, ac("router", "fast"));
    arrow(s, m("router").r, m("router").cy + 0.12, N["expert"].x, m("expert").cy, ac("router", "expert"));
    if (stage >= 5) {
      // lanes -> trust -> answer / human
      arrow(s, m("fast").r, m("fast").cy, N["trust"].x, m("trust").cy - 0.12, ac("trust"));
      arrow(s, m("expert").r, m("expert").cy, N["trust"].x, m("trust").cy + 0.12, ac("trust"));
      arrow(s, m("trust").r, m("trust").cy - 0.12, N["ans"].x, m("ans").cy, ac("trust"));
      arrow(s, m("trust").r, m("trust").cy + 0.12, N["human"].x, m("human").cy, ac("trust", "human"));
    } else {
      arrow(s, m("fast").r, m("fast").cy, N["ans"].x, m("ans").cy, DIM_EDGE);
      arrow(s, m("expert").r, m("expert").cy, N["ans"].x, m("ans").cy + 0.1, DIM_EDGE);
    }
  }
  if (stage >= 4) {
    // fabric feeds the model column
    arrow(s, 8.35, N["fabric"].y, 8.35, N["expert"].y + BH + 0.02, ac("fabric"));
  }
  if (stage >= 6) {
    // Answer -> (around the right and under the row) -> Ground truth
    arrow(s, 12.95, m("ans").b, 12.95, 6.86, RED, 1.25, false);
    arrow(s, 12.95, 6.86, 3.1, 6.86, RED, 1.25, false);
    arrow(s, 3.1, 6.86, 3.1, m("gt").b + 0.02, RED, 1.25);
    // Ground truth -> cache -> retrain (through the gaps)
    arrow(s, m("gt").r, m("gt").cy, N["cache"].x, m("cache").cy, RED, 1.25);
    arrow(s, m("cache").r, m("cache").cy, N["retrain"].x, m("retrain").cy, RED, 1.25);
    // retrain -> expert lane, routed right of the fabric and below the trust gate
    arrow(s, m("retrain").r, m("retrain").cy, 10.7, m("retrain").cy, RED, 1.25, false);
    arrow(s, 10.7, m("retrain").cy, 10.7, 3.9, RED, 1.25, false);
    arrow(s, 10.7, 3.9, m("expert").r + 0.02, 3.9, RED, 1.25);
  }

  // caption
  const [t, c] = STAGE_META[stage];
  s.addText([
    { text: t + "   ", options: { color: RED, bold: true, fontSize: 13 } },
    { text: c, options: { color: TEXT, fontSize: 13 } },
  ], { x: 0.55, y: 6.95, w: 12.2, h: 0.5, fontFace: BODY, valign: "top" });
}

// ═════════════════════════════ SLIDE — PROOF ═════════════════════════════════
function slideProof() {
  const s = newSlide();
  kicker(s, "MODEL INSIGHTS · MEASURED ON 18 HELD-OUT, PARAPHRASED QUESTIONS");
  s.addText("Proof, not promises", { x: 0.5, y: 0.78, w: 12.3, h: 0.7, fontFace: HEAD, fontSize: 30, color: TEXT, bold: true });

  s.addChart(pptx.ChartType.bar, [
    { name: "Base Qwen3-14B", labels: ["TOTAL", "Public telecom", "Hardware", "Error codes", "Billing codes"], values: [22, 80, 0, 0, 0] },
    { name: "TruthLine fine-tuned", labels: ["TOTAL", "Public telecom", "Hardware", "Error codes", "Billing codes"], values: [94, 100, 100, 100, 80] },
  ], {
    x: 0.55, y: 1.75, w: 6.4, h: 4.4,
    barDir: "bar", barGapWidthPct: 60, barOverlapPct: -15,
    chartColors: ["6E6A63", "ED1C24"],
    showLegend: true, legendPos: "b", legendColor: MUTED, legendFontSize: 11,
    catAxisLabelColor: TEXT, catAxisLabelFontSize: 12,
    valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
    valAxisMaxVal: 100, valAxisMinVal: 0, valAxisMajorUnit: 25,
    valGridLine: { color: "2A2A2A", style: "solid", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true, dataLabelColor: TEXT, dataLabelFontSize: 10, dataLabelPosition: "outEnd",
    plotArea: { fill: { color: BG } }, chartArea: { fill: { color: BG } },
  });
  s.addText("Proprietary facts are synthetic — invented by us — so the base model cannot know them. The improvement is provable, not anecdotal.",
    { x: 0.55, y: 6.3, w: 6.4, h: 0.8, fontFace: BODY, fontSize: 11.5, color: MUTED, italic: true });

  const cards = [
    ["60 s", "full LoRA retrain on one MI300X — r=32, bf16, ~0.9% of parameters trained"],
    ["−74%", "tokens per answer (302 → 78) · −51% end-to-end latency"],
    ["<35%", "of one 192 GB MI300X hosts the entire serving stack: 14B + 1.5B + comparison copy"],
    ["98%", "GPU utilization at ~740 W during training — live rocm-smi telemetry built into the product"],
  ];
  cards.forEach(([num, label], i) => {
    const x = 7.35 + (i % 2) * 2.85, y = 1.75 + Math.floor(i / 2) * 2.3;
    card(s, x, y, 2.65, 2.1);
    s.addText(num, { x: x + 0.2, y: y + 0.18, w: 2.25, h: 0.75, fontFace: HEAD, fontSize: 32, color: RED, bold: true });
    s.addText(label, { x: x + 0.2, y: y + 0.98, w: 2.3, h: 1.05, fontFace: BODY, fontSize: 11, color: TEXT });
  });
}

// ═════════════════════════════ SLIDE — IMPACT ════════════════════════════════
function slideImpact() {
  const s = newSlide();
  kicker(s, "IMPACT · DEMO · ROAD AHEAD");
  s.addText("Built this week. Ready for Monday morning.", { x: 0.5, y: 0.78, w: 12.3, h: 0.7, fontFace: HEAD, fontSize: 27, color: TEXT, bold: true });

  // Shipped column
  card(s, 0.55, 1.7, 4.6, 4.45, { edge: GREEN, edgeW: 1.25 });
  s.addText("SHIPPED — NOT ROADMAP", { x: 0.85, y: 1.88, w: 4.0, h: 0.3, fontFace: BODY, fontSize: 12, color: GREEN, bold: true, charSpacing: 2 });
  const shipped = [
    "Fine-tuned 14B + 1.5B domain-expert models",
    "7-stage agentic pipeline — guardrails, clarity gate, trust gate",
    "MCP enterprise server: expert escalation routing + live outage feed",
    "Hybrid RAG (BM25 + vector, RRF query fusion)",
    "Tier-zero semantic cache + data flywheel",
    "vLLM serving path · 4-tab console · live AMD telemetry",
    "Held-out evaluation harness (the 94% number)",
  ];
  s.addText(shipped.map(t => ({ text: t, options: { bullet: { code: "2713", indent: 14 }, color: TEXT, breakLine: true } })),
    { x: 0.85, y: 2.25, w: 4.05, h: 3.8, fontFace: BODY, fontSize: 12, lineSpacing: 19, valign: "top" });

  // Demo column
  card(s, 5.4, 1.7, 3.6, 4.45);
  s.addText("THE DEMO", { x: 5.7, y: 1.88, w: 3.0, h: 0.3, fontFace: BODY, fontSize: 12, color: RED, bold: true, charSpacing: 2 });
  const demo = [
    "Base model hallucinates live — TruthLine answers from its weights",
    "PII masked in the pipeline trace",
    "MCP tool calls visible in a real server terminal",
    "👍 → instant zero-GPU cache hit",
    "GPU lights up for a 60-second retrain",
  ];
  s.addText(demo.map(t => ({ text: t, options: { bullet: { indent: 14 }, color: TEXT, breakLine: true } })),
    { x: 5.7, y: 2.25, w: 3.1, h: 3.8, fontFace: BODY, fontSize: 12, lineSpacing: 19, valign: "top" });

  // Road ahead column
  card(s, 9.3, 1.7, 3.5, 4.45);
  s.addText("SCALING FROM HERE", { x: 9.6, y: 1.88, w: 3.0, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTED, bold: true, charSpacing: 2 });
  const road = [
    "Batched vLLM serving under production load",
    "MCP connectors into live CRM & billing",
    "GRPO / DPO on rejection signals",
    "Corpus scaled with anonymized transcripts — every adapter eval-gated",
  ];
  s.addText(road.map(t => ({ text: t, options: { bullet: { indent: 14 }, color: TEXT, breakLine: true } })),
    { x: 9.6, y: 2.25, w: 3.0, h: 3.8, fontFace: BODY, fontSize: 12, lineSpacing: 19, valign: "top" });

  // closing banner
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.55, y: 6.45, w: 12.25, h: 0.72, rectRadius: 0.07, fill: { color: RED_DARK }, line: { color: RED, width: 1 } });
  s.addText("TruthLine doesn't just answer correctly today — every interaction makes tomorrow's model better, for one minute of AMD GPU time a night.",
    { x: 0.85, y: 6.49, w: 11.7, h: 0.64, fontFace: BODY, fontSize: 14.5, color: "FFFFFF", bold: true, valign: "middle" });
}

// ── Assemble ─────────────────────────────────────────────────────────────────
slideTitle();
slideProblem();
if (MODE === "animated") {
  for (let st = 1; st <= 6; st++) slideStage(st);
} else {
  slideStage(6);
}
slideProof();
slideImpact();

pptx.writeFile({ fileName: OUT }).then(() => console.log("written:", OUT));
