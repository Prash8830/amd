"""Model Router — right-sizes the model per query.

Simple, high-confidence FAQ queries go to the fast small model; anything
involving proprietary codes, hardware troubleshooting, or low routing
confidence goes to the 14B expert. On the MI300X both live in VRAM at once
(~31 GB of 192), so routing trades nothing and saves latency + GPU cycles.
"""

from __future__ import annotations
import re
from dataclasses import dataclass

CODE_PATTERN = re.compile(r"\b[A-Z]{1,4}-\d{2,4}[A-Za-z]*\b")
EXPERT_TERMS = ("troubleshoot", "step", "error", "fail", "los ", "blink",
                "flash", "reset", "pair", "activat", "provision")


@dataclass
class RouteDecision:
    route: str          # "fast" | "expert"
    reason: str


class ModelRouter:
    def route(self, query: str, intent_confidence: float) -> RouteDecision:
        ql = query.lower()

        if CODE_PATTERN.search(query):
            return RouteDecision("expert", "proprietary code/hardware reference")
        if any(t in ql for t in EXPERT_TERMS):
            return RouteDecision("expert", "troubleshooting language")
        if intent_confidence < 0.6:
            return RouteDecision("expert", f"low routing confidence ({intent_confidence:.2f})")
        if len(ql.split()) > 25:
            return RouteDecision("expert", "long multi-part query")

        return RouteDecision("fast", f"simple FAQ, confidence {intent_confidence:.2f}")
