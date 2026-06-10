"""Intent Classifier Agent — classifies telecom support queries."""

from __future__ import annotations
import re
from dataclasses import dataclass

INTENTS = {
    "billing": ["bill", "billing", "charge", "payment", "invoice", "autopay", "fee", "refund", "overcharge", "due", "balance", "statement", "credit"],
    "network": ["signal", "coverage", "slow", "speed", "5g", "4g", "lte", "connection", "wifi", "roaming", "data", "volte", "drop", "fiber", "ont", "los", "outage", "voicemail", "internet"],
    "device": ["phone", "device", "unlock", "screen", "battery", "hardware", "broken", "repair", "warranty", "transfer", "router", "modem", "gateway", "mesh", "extender", "sim", "esim", "activating", "activation"],
    "plan": ["plan", "upgrade", "downgrade", "unlimited", "international", "add line", "change", "package"],
    "account": ["account", "password", "login", "profile", "cancel", "port", "number", "email", "address", "settings"],
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched_keywords: list[str]


class IntentClassifierAgent:
    """Keyword-based intent classifier with confidence scoring."""

    def classify(self, query: str) -> IntentResult:
        query_lower = query.lower()
        scores: dict[str, list[str]] = {}

        for intent, keywords in INTENTS.items():
            matched = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', query_lower)]
            if matched:
                scores[intent] = matched

        if not scores:
            return IntentResult(intent="general", confidence=0.3, matched_keywords=[])

        best_intent = max(scores, key=lambda k: len(scores[k]))
        total_keywords = sum(len(v) for v in INTENTS.values())
        confidence = min(0.95, 0.5 + len(scores[best_intent]) / total_keywords * 10)

        return IntentResult(
            intent=best_intent,
            confidence=round(confidence, 2),
            matched_keywords=scores[best_intent],
        )
