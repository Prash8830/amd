/* TruthLine — TCS x AMD submission deck (designed, logo-intact).
   Answers the template questions in order, clean layout. node make_submission_deck.js */
const P = require("pptxgenjs");
const D = "/tmp/deckbuild/";
const p = new P();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";

// palette
const INK="1A1A1A", MUT="6B6B6B", RED="ED1C24", BG="FFFFFF", CARD="F5F4F1",
      EDGE="E3E1DB", TEAL="0F6E56", GREEN="2FBF71";
const HEAD="Georgia", BODY="Calibri";
const LOGO=D+"logo_tcs_amd.png", TATA=D+"logo_tata.png", BGIMG=D+"title_bg.jpg";

function header(s, kicker, title){
  s.background={color:BG};
  s.addImage({path:LOGO, x:0.45, y:0.32, w:1.7, h:0.62});
  s.addImage({path:TATA, x:12.5, y:0.32, w:0.46, h:0.43});
  s.addText(kicker,{x:0.5,y:1.06,w:12.3,h:0.28,fontFace:BODY,fontSize:11,color:RED,bold:true,charSpacing:3});
  s.addText(title,{x:0.5,y:1.30,w:12.3,h:0.62,fontFace:HEAD,fontSize:30,color:INK,bold:true});
}
function card(s,x,y,w,h,label,body,opts={}){
  s.addShape(p.shapes.ROUNDED_RECTANGLE,{x,y,w,h,rectRadius:0.06,
    fill:{color:opts.fill||CARD},line:{color:opts.edge||EDGE,width:1}});
  s.addText(label,{x:x+0.22,y:y+0.16,w:w-0.44,h:0.34,fontFace:BODY,fontSize:12,
    color:opts.lc||RED,bold:true,charSpacing:1});
  s.addText(body,{x:x+0.22,y:y+0.54,w:w-0.44,h:h-0.7,fontFace:BODY,
    fontSize:opts.bs||12.5,color:opts.bc||INK,valign:"top",lineSpacingMultiple:1.02});
}

// ───────────────────────── SLIDE 1 — TITLE (butterfly bg) ─────────────────────
let s=p.addSlide(); s.background={color:"06080D"};
s.addImage({path:BGIMG,x:0,y:0,w:13.333,h:7.5});
s.addShape(p.shapes.ROUNDED_RECTANGLE,{x:0.4,y:0.4,w:2.5,h:1.0,rectRadius:0.08,fill:{color:"FFFFFF"},line:{type:"none"}});
s.addImage({path:LOGO,x:0.6,y:0.55,w:2.1,h:0.76});
s.addText("TCS & AMD AI Hackathon",{x:0.6,y:2.5,w:7.0,h:0.5,fontFace:BODY,fontSize:16,color:"CADCFC",charSpacing:2});
s.addText([{text:"Truth",options:{color:"FFFFFF"}},{text:"Line",options:{color:RED}}],
  {x:0.55,y:3.0,w:8,h:1.2,fontFace:HEAD,fontSize:60,bold:true});
s.addText("Telco-Specific Customer LLM — a hallucination-resistant, fine-tuned support model on AMD MI300X",
  {x:0.6,y:4.25,w:7.2,h:0.8,fontFace:BODY,fontSize:16,color:"E8Eaee"});
s.addText("Track 3 · Fine-Tuning · FINETUNING_002      |      Prashant Patil",
  {x:0.6,y:6.5,w:9,h:0.4,fontFace:BODY,fontSize:13,color:"AEB6C2"});

// ───────────────────────── SLIDE 2 — BASIC INFORMATION ───────────────────────
s=p.addSlide(); header(s,"BASIC INFORMATION","TruthLine — Telco-Specific Customer LLM");
card(s,0.5,2.15,6.05,1.5,"TEAM NAME","Prashant Patil   (Emp ID: 2437783)");
card(s,6.75,2.15,6.05,1.5,"MEMBER / ROLE",
  "Prashant Patil — solo participant. Research, dataset curation, fine-tuning, multi-agent engineering, evaluation, business case & presentation.");
card(s,0.5,3.85,12.3,1.35,"PROJECT TITLE",
  "TruthLine — a domain-expert telecom support LLM: proprietary knowledge in the weights, facts in the knowledge fabric, tools on the protocol, humans in the loop.");
card(s,0.5,5.40,12.3,1.75,"SHORT DESCRIPTION",
  "A domain-expert telecom support LLM. We fine-tuned Qwen3-14B on AMD Instinct MI300X so proprietary knowledge — internal billing codes, router hardware, error codes — lives in the model's weights, wrapped in an agentic pipeline: guardrails, clarity gate, semantic cache, hybrid RAG, model router, trust gate, MCP enterprise tools, and a data flywheel. Result: held-out accuracy 22% → 94%, at 74% fewer tokens per answer — on one AMD GPU, with zero external API calls.");

// ───────────────────────── SLIDE 3 — PROBLEM & CONTEXT ───────────────────────
s=p.addSlide(); header(s,"PROBLEM & CONTEXT","Generic LLMs hallucinate on proprietary telecom knowledge");
card(s,0.5,2.15,12.3,1.7,"PROBLEM STATEMENT  (use case FINETUNING_002)",
  "Generic public LLMs hallucinate when faced with proprietary telecom jargon, specific router hardware models, and complex internal billing codes. Fine-tuning an open-source model on years of anonymized telecom support transcripts and technical manuals to create a domain-expert support model that provides perfectly accurate, step-by-step troubleshooting.",
  {bs:13.5});
card(s,0.5,4.05,4.0,3.0,"TARGET USERS / STAKEHOLDERS",
  "Telecom contact centres — customer self-service and agent-assist.\n\nBroadly: any enterprise whose vocabulary (codes, SKUs, procedures) isn't on the public internet.\n\nStakeholders: support agents, NOC/operations, customers, support-cost owners.");
card(s,4.65,4.05,4.0,3.0,"WHY IT MATTERS",
  "Support is high-volume and policy-bound; every confident wrong answer becomes a complaint, escalation, or churn — risk manufactured by the AI itself.\n\nAccurate step-by-step answers deflect tickets and cut handle time, at 74% fewer tokens (~4x throughput per GPU).");
card(s,8.8,4.05,4.0,3.0,"MAPPED HACKATHON CHALLENGE",
  "Track 3 — Fine-Tuning (Advanced).\n\nUse case FINETUNING_002: Telco-Specific Customer LLM.\n\nPEFT / LoRA fine-tuning with measurable, reproducible accuracy and efficiency gains over the base model.");

// ───────────────────────── SLIDE 4 — SOLUTION OVERVIEW ───────────────────────
s=p.addSlide(); header(s,"SOLUTION OVERVIEW","Eight-stage agentic pipeline, fully on AMD MI300X");

// --- architecture diagram band (native shapes) ---
const dy=2.05;
function node(x,y,w,h,t,sub,col){
  const fill = col==="red"?"FBE3E4":col==="teal"?"E1F5EE":col==="gray"?"EFEFEC":"F5F4F1";
  const edge = col==="red"?RED:col==="teal"?TEAL:EDGE;
  s.addShape(p.shapes.ROUNDED_RECTANGLE,{x,y,w,h,rectRadius:0.05,fill:{color:fill},line:{color:edge,width:1}});
  s.addText([{text:t+(sub?"\n":""),options:{fontSize:10.5,bold:true,color:INK}},
             {text:sub||"",options:{fontSize:8.5,color:MUT}}],
    {x:x+0.04,y:y+0.03,w:w-0.08,h:h-0.06,align:"center",valign:"middle",fontFace:BODY,lineSpacingMultiple:0.95});
}
function arr(x1,y1,x2,y2,c){s.addShape(p.shapes.LINE,{x:Math.min(x1,x2),y:Math.min(y1,y2),
  w:Math.abs(x2-x1)||0.001,h:Math.abs(y2-y1)||0.001,flipH:x2<x1,flipV:y2<y1,
  line:{color:c||"9A968E",width:1.25,endArrowType:"triangle"}});}

// One continuous left-to-right pipeline. Two columns are stacked pairs
// (the model lanes and the two outcomes), so nothing wraps or overflows.
const yMid=dy+0.55, bw=1.18, gap=0.10, bh=0.72;
function single(x,t,sub,col){ node(x,yMid-bh/2,bw,bh,t,sub,col); }
function pair(x,tA,sA,tB,sB,colA,colB){
  node(x,yMid-0.54,bw,0.5,tA,sA,colA); node(x,yMid+0.04,bw,0.5,tB,sB,colB);
}
let cx=0.2;
const cols=[
 ["s","Customer query","","gray"],
 ["s","Guardrails","PII·inject","red"],
 ["s","Clarity gate","ask first","red"],
 ["s","Semantic cache","0-GPU","teal"],
 ["s","Intent","classify","gray"],
 ["s","Hybrid RAG","BM25+vec·RRF","red"],
 ["s","Model router","right-size","red"],
 ["p","Fast 1.5B","FAQ","Expert 14B","codes","teal","teal"],
 ["s","Trust gate","<0.6 esc.","red"],
 ["p","Answer","trust OK","Human+MCP","ticket","gray","red"],
];
const centers=[];
cols.forEach((c,i)=>{
  if(i){arr(cx-gap,yMid,cx,yMid);}
  centers.push(cx+bw/2);
  if(c[0]==="s") single(cx,c[1],c[2],c[3]);
  else pair(cx,c[1],c[2],c[3],c[4],c[5],c[6]);
  cx+=bw+gap;
});
// AMD callout (full width, below pipeline) + flywheel line
node(0.2,dy+1.55,cx-0.3,0.5,"AMD Instinct MI300X · 192 GB · ROCm",
     "both fine-tuned lanes trained (~60 s) and served on one card · live rocm-smi telemetry","teal");
s.addText("Data flywheel:  approved answers → ground-truth store → semantic cache (instant, 0-GPU) + ~60-second LoRA retrain → smarter weights, nightly",
  {x:0.2,y:dy+2.12,w:12.9,h:0.3,fontFace:BODY,fontSize:10.5,italic:true,color:TEAL,align:"center"});

// --- 3-column strip ---
const cy=4.55, cw=4.0;
card(s,0.5,cy,cw,2.55,"AI APPROACH",
  "Fine-tuning (PEFT / LoRA) embeds proprietary domain knowledge in the weights; RAG grounds facts that change. Agentic orchestration on LangGraph with MCP enterprise tools and a feedback-to-weights data flywheel.",{bs:11.5});
card(s,4.65,cy,cw,2.55,"KEY TECHNOLOGIES",
  "Qwen3-14B + Qwen2.5-1.5B (open weights) · HuggingFace transformers + PEFT/LoRA (bf16) · LangGraph · LangChain hybrid RAG (BM25 + ChromaDB, RRF) · MCP SDK · vLLM · Streamlit · FastAPI · rocm-smi. AMD MI300X / ROCm — zero external API calls.",{bs:11.5});
card(s,8.8,cy,cw,2.55,"BUILT DURING HACKATHON",
  "All of it, from scratch: curated corpus + invented proprietary layer; LoRA fine-tuning of 14B + 1.5B; the LangGraph pipeline; guardrails, cache, router, trust gate; MCP server (expert routing, ITSM, outage feed); hybrid RAG; data flywheel; eval harness; 5-tab AMD console.",{bs:11.5});

// ───────────────────────── SLIDE 5 — MODEL INSIGHTS ──────────────────────────
s=p.addSlide(); header(s,"MODEL INSIGHTS","22% → 94% accuracy · 74% fewer tokens · ~60s retrain");
// stat tiles
function stat(x,y,w,num,lab){
  s.addShape(p.shapes.ROUNDED_RECTANGLE,{x,y,w,h:1.4,rectRadius:0.06,fill:{color:"111111"},line:{color:"111111",width:1}});
  s.addText(num,{x:x+0.15,y:y+0.18,w:w-0.3,h:0.6,fontFace:HEAD,fontSize:26,bold:true,color:"FFFFFF"});
  s.addText(lab,{x:x+0.15,y:y+0.82,w:w-0.3,h:0.5,fontFace:BODY,fontSize:10.5,color:"C9C7C0"});
}
stat(0.5,2.15,3.0,"22% → 94%","held-out accuracy (18 paraphrased Qs)");
stat(3.65,2.15,3.0,"−74%","tokens/answer (302 → 78) · −51% latency");
stat(6.8,2.15,3.0,"~60 s","full LoRA retrain on one MI300X");
stat(9.95,2.15,2.85,"98% @ 740W","GPU utilization (rocm-smi)");
// cards
card(s,0.5,3.75,4.0,3.35,"MODELS USED",
  "Qwen/Qwen3-14B — expert lane (LoRA, adapter-merged).\nQwen/Qwen2.5-1.5B — fast lane.\nBase Qwen3-14B kept loadable for live hallucination comparison.\nEmbeddings: all-MiniLM-L6-v2.",{bs:12});
card(s,4.65,3.75,4.0,3.35,"DATASET (FINE-TUNING)",
  "168 synthetic instruction-response samples (68 unique answers), incl. a proprietary layer of 24 invented internal facts — billing codes, CPE hardware, error codes.\n\nInvented deliberately so they're absent from any base model's pretraining, making the accuracy gain provable. Production swaps in anonymized transcripts.",{bs:12});
card(s,8.8,3.75,4.0,3.35,"TRAINING · TOKENS · LATENCY · GPU",
  "Training: LoRA bf16, rank 32, 12 epochs, ~0.9% of params; ~60s (14B), ~55s (1.5B).\nTokens: 78/answer fine-tuned vs 302 base (B-204: 66 vs 320-cap).\nLatency: cache ~10ms · 1.5B ~1-2s · 14B ~4-13s.\nGPU: 192GB MI300X; full stack <35% of one card.",{bs:11});

// ───────────────────────── SLIDE 6 — IMPACT & DEMO ───────────────────────────
s=p.addSlide(); header(s,"IMPACT & DEMO SUMMARY","Trusted deflection that gets smarter every night");
card(s,0.5,2.15,6.05,2.55,"EXPECTED IMPACT / VALUE",
  "22% → 94% accuracy and 74% fewer tokens (~4x throughput per GPU).\n\nLow-trust answers escalate to the right on-call expert via MCP with an auto ITSM ticket — never reaching the customer. Repeat questions become zero-GPU cache hits, and the model improves nightly from human feedback in ~60s of GPU time, with no ML team in the loop.",{bs:12});
card(s,6.75,2.15,6.05,2.55,"KEY DIFFERENTIATORS / INNOVATION",
  "1. Measurable fine-tuning — synthetic proprietary eval makes 22%→94% provable.\n2. Data flywheel — approvals become a zero-GPU cache AND the next training set (viable only because retraining costs ~60s on AMD).\n3. Right-sized compute — cache / 1.5B / 14B / human tiers on one card.\n4. Self-policing trust gate; fully local on AMD ROCm.",{bs:11.5});
card(s,0.5,4.9,12.3,2.2,"DEMO FLOW — LIVE HIGHLIGHTS",
  "1. Base vs fine-tuned, live — base invents internal code B-204; TruthLine answers correctly from its weights.   2. Pipeline trace — PII masked, model-router decision, live MCP outage feed, trust score.   3. Right-sizing — FAQ → 1.5B fast lane, proprietary codes → 14B expert.   4. Flywheel — thumbs-up an answer, re-ask → instant zero-GPU cache hit.   5. Self-policing — an invented price trips the trust gate → escalation + MCP ITSM ticket.   6. Observability — live rocm-smi telemetry; a ~60-second retrain at 98% GPU.   Result: 22% → 94% accuracy, −74% tokens.",{bs:12.5});

// ═══════════ APPENDED: animated architecture build (6 stages, dark) ═══════════
(function appendArchitecture(){
  const gBG="121212",gCARD="1D1D1D",gEDGE="3A3A3A",gTEXT="F4F1EA",gMUT="A9A399",
        gRED="ED1C24",gDIM="4A4A4A",gHEAD="Arial Black",gBODY="Calibri";
  const ROW=2.5,BH=0.95,RTOP=1.62,RBOT=3.42;
  const N0=[
    {id:"cust",label:"Customer\nquery",x:0.5,y:ROW,w:1.32,stage:1},
    {id:"llm",label:"Generic LLM",x:5.6,y:ROW,w:1.9,stage:1,only:1},
    {id:"ft",label:"Domain-expert\nmodel (1.5B)",x:5.6,y:ROW,w:1.9,stage:2,only:2},
    {id:"guard",label:"Guardrails\nPII · injection",x:2.07,y:ROW,w:1.45,stage:5},
    {id:"clarity",label:"Clarity gate\nask, don't guess",x:3.77,y:ROW,w:1.5,stage:5},
    {id:"router",label:"Model router\nright-sized",x:5.6,y:ROW,w:1.45,stage:3},
    {id:"fast",label:"Fast lane\n1.5B fine-tuned",x:7.45,y:RTOP,w:1.8,stage:3},
    {id:"expert",label:"Expert lane\n14B · vLLM",x:7.45,y:RBOT,w:1.8,stage:3},
    {id:"trust",label:"Trust gate\nscore < 0.6 ↗",x:9.65,y:ROW,w:1.45,stage:5},
    {id:"ans",label:"Answer",x:11.45,y:RTOP,w:1.7,stage:1},
    {id:"human",label:"On-call expert\nvia MCP",x:11.45,y:RBOT,w:1.7,stage:5},
    {id:"fabric",label:"Knowledge fabric — hybrid RAG (BM25 + vector, RRF) · MCP outage feed",x:4.35,y:4.85,w:6.0,stage:4},
    {id:"gt",label:"Ground-truth DB\napproved pairs",x:2.0,y:5.95,w:2.2,h:0.8,stage:6},
    {id:"cache",label:"Tier-zero cache\n0 GPU · 10 ms",x:5.0,y:5.95,w:2.1,h:0.8,stage:6},
    {id:"retrain",label:"60 s LoRA retrain\non MI300X",x:7.9,y:5.95,w:2.2,h:0.8,stage:6},
  ];
  const META=[null,
    ["WHERE EVERYONE STARTS","Customer query → LLM → answer. This is where hallucination lives — the model answers whether it knows or not."],
    ["DOMAIN TRUTH IN THE WEIGHTS","A 60-second LoRA fine-tune turns the generic LLM into a domain-expert model. B-204 now lives in the weights."],
    ["RIGHT-SIZED COMPUTE","A router sends simple FAQs to the 1.5B fast lane, proprietary codes to the 14B expert lane — in-process or served via vLLM. Never a bulldozer for a thumbtack."],
    ["FACTS IN THE KNOWLEDGE FABRIC","Hybrid retrieval (BM25 + vector, RRF query fusion) grounds every answer; live enterprise data arrives over our MCP server."],
    ["DUTY OF CARE","Guardrails mask PII and block injections; the clarity gate asks instead of guessing; the trust gate routes low-trust answers via MCP to the on-call domain expert — never to the customer."],
    ["THE DATA FLYWHEEL","Every thumbs-up becomes ground truth: served instantly from the tier-zero cache (zero GPU) and auto-merged into the next 60-second retrain. Every approval becomes tomorrow's weights."],
  ];
  const gNew=()=>{const s=p.addSlide(); s.background={color:gBG}; return s;};
  const gKick=(s,t)=>{ s.addShape(p.shapes.RECTANGLE,{x:0.55,y:0.47,w:0.14,h:0.14,fill:{color:gRED},line:{type:"none"}});
    s.addText(t,{x:0.79,y:0.36,w:11,h:0.34,fontFace:gBODY,fontSize:11.5,color:gMUT,bold:true,charSpacing:3}); };
  const gCard=(s,x,y,w,h,o={})=>{ s.addShape(p.shapes.ROUNDED_RECTANGLE,{x,y,w,h,rectRadius:0.07,
    fill:{color:o.fill||gCARD},line:{color:o.edge||gEDGE,width:o.edgeW||1}}); };
  const mid=n=>{const h=n.h||BH; return {cx:n.x+n.w/2,cy:n.y+h/2,r:n.x+n.w,t:n.y,b:n.y+h};};
  const gArr=(s,x1,y1,x2,y2,c,w=1.5,head=true)=>{ s.addShape(p.shapes.LINE,{x:Math.min(x1,x2),y:Math.min(y1,y2),
    w:Math.abs(x2-x1)||0.001,h:Math.abs(y2-y1)||0.001,flipH:x2<x1,flipV:y2<y1,
    line:head?{color:c,width:w,endArrowType:"triangle"}:{color:c,width:w}}); };
  const gDraw=(s,n,stage)=>{ const isNew=(n.stage===stage); if(n.stage>stage)return;
    const nh=n.id==="fabric"?0.7:(n.h||BH);
    gCard(s,n.x,n.y,n.w,nh,{edge:isNew?gRED:gEDGE,edgeW:isNew?2:1});
    const pr=n.label.split("\n");
    if(pr[1]!==undefined && n.id!=="fabric"){
      s.addText([{text:pr[0]+"\n",options:{bold:true,fontSize:12.5,color:gTEXT}},
                 {text:pr[1],options:{fontSize:10,color:isNew?"FFB3B6":gMUT}}],
        {x:n.x+0.06,y:n.y+0.06,w:n.w-0.12,h:BH-0.12,fontFace:gBODY,align:"center",valign:"middle",lineSpacing:13});
    } else {
      s.addText(n.label,{x:n.x+0.1,y:n.y+0.05,w:n.w-0.2,h:nh-0.1,fontFace:gBODY,
        fontSize:n.id==="fabric"?12:12.5,bold:true,color:gTEXT,align:"center",valign:"middle"});
    }
  };
  function stageSlide(stage){
    const s=gNew(); gKick(s,`ARCHITECTURE · STEP ${stage} OF 6`);
    s.addText("Six decisions that kill hallucination",{x:0.5,y:0.72,w:12.3,h:0.6,fontFace:gHEAD,fontSize:25,color:gTEXT,bold:true});
    const N=Object.fromEntries(N0.map(n=>[n.id,(n.id==="ans"&&stage<=2)?{...n,y:ROW}:n]));
    const ac=(...ids)=>ids.some(id=>N[id].stage===stage)?gRED:gDIM;
    Object.values(N).forEach(n=>{ if(!n.only||n.only===stage) gDraw(s,n,stage); });
    const m=id=>mid(N[id]);
    if(stage<=2){ const md=stage===1?"llm":"ft";
      gArr(s,m("cust").r,m("cust").cy,N[md].x,m(md).cy,ac("cust",md));
      gArr(s,m(md).r,m(md).cy,N["ans"].x,ROW+BH/2,ac(md));
    } else {
      if(stage>=5){
        gArr(s,m("cust").r,m("cust").cy,N["guard"].x,m("guard").cy,ac("guard"));
        gArr(s,m("guard").r,m("guard").cy,N["clarity"].x,m("clarity").cy,ac("guard","clarity"));
        gArr(s,m("clarity").r,m("clarity").cy,N["router"].x,m("router").cy,ac("clarity"));
      } else { gArr(s,m("cust").r,m("cust").cy,N["router"].x,m("router").cy,gDIM); }
      gArr(s,m("router").r,m("router").cy-0.12,N["fast"].x,m("fast").cy,ac("router","fast"));
      gArr(s,m("router").r,m("router").cy+0.12,N["expert"].x,m("expert").cy,ac("router","expert"));
      if(stage>=5){
        gArr(s,m("fast").r,m("fast").cy,N["trust"].x,m("trust").cy-0.12,ac("trust"));
        gArr(s,m("expert").r,m("expert").cy,N["trust"].x,m("trust").cy+0.12,ac("trust"));
        gArr(s,m("trust").r,m("trust").cy-0.12,N["ans"].x,m("ans").cy,ac("trust"));
        gArr(s,m("trust").r,m("trust").cy+0.12,N["human"].x,m("human").cy,ac("trust","human"));
      } else {
        gArr(s,m("fast").r,m("fast").cy,N["ans"].x,m("ans").cy,gDIM);
        gArr(s,m("expert").r,m("expert").cy,N["ans"].x,m("ans").cy+0.1,gDIM);
      }
    }
    if(stage>=4){ gArr(s,8.35,N["fabric"].y,8.35,N["expert"].y+BH+0.02,ac("fabric")); }
    if(stage>=6){
      gArr(s,12.95,m("ans").b,12.95,6.86,gRED,1.25,false);
      gArr(s,12.95,6.86,3.1,6.86,gRED,1.25,false);
      gArr(s,3.1,6.86,3.1,m("gt").b+0.02,gRED,1.25);
      gArr(s,m("gt").r,m("gt").cy,N["cache"].x,m("cache").cy,gRED,1.25);
      gArr(s,m("cache").r,m("cache").cy,N["retrain"].x,m("retrain").cy,gRED,1.25);
      gArr(s,m("retrain").r,m("retrain").cy,10.7,m("retrain").cy,gRED,1.25,false);
      gArr(s,10.7,m("retrain").cy,10.7,3.9,gRED,1.25,false);
      gArr(s,10.7,3.9,m("expert").r+0.02,3.9,gRED,1.25);
    }
    const meta=META[stage];
    s.addText([{text:meta[0]+"   ",options:{color:gRED,bold:true,fontSize:13}},
               {text:meta[1],options:{color:gTEXT,fontSize:13}}],
      {x:0.55,y:6.95,w:12.2,h:0.5,fontFace:gBODY,valign:"top"});
  }
  for(let st=1; st<=6; st++) stageSlide(st);
})();

// ───────────────────────── SLIDE — THANK YOU (closer) ────────────────────────
s=p.addSlide(); s.background={color:"06080D"};
s.addImage({path:BGIMG,x:0,y:0,w:13.333,h:7.5});
s.addShape(p.shapes.ROUNDED_RECTANGLE,{x:0.4,y:0.4,w:2.5,h:1.0,rectRadius:0.08,fill:{color:"FFFFFF"},line:{type:"none"}});
s.addImage({path:LOGO,x:0.6,y:0.55,w:2.1,h:0.76});
s.addShape(p.shapes.ROUNDED_RECTANGLE,{x:12.25,y:0.4,w:0.85,h:0.85,rectRadius:0.08,fill:{color:"FFFFFF"},line:{type:"none"}});
s.addImage({path:TATA,x:12.45,y:0.5,w:0.45,h:0.43});
s.addText("Thank you",{x:0.6,y:3.0,w:8,h:1,fontFace:HEAD,fontSize:48,bold:true,color:"FFFFFF"});
s.addText([{text:"Truth",options:{color:"FFFFFF"}},{text:"Line",options:{color:RED}},
           {text:"  —  domain truth in the weights, on AMD.",options:{color:"CADCFC"}}],
  {x:0.6,y:4.1,w:11,h:0.6,fontFace:BODY,fontSize:18});

p.writeFile({fileName:"/Users/prashant/Documents/TCS_AMD_AI_Hackathon_Submission_TruthLine.pptx"})
 .then(f=>console.log("written:",f));
