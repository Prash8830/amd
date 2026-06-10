"""Guardrails Agent — input/output safety layer for the support pipeline.

Input side:  PII masking (phone, email, card, account numbers) before the
             query reaches the LLM or logs; prompt-injection screening.
Output side: PII leak detection; flags dollar amounts that don't appear in
             the grounding context (possible hallucinated pricing).

Telecom support is a regulated, PII-heavy domain — in production this layer
is mandatory, not optional. Detections are surfaced in the dashboard's
pipeline trace for auditability.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

# Order matters: longest/most specific first so card numbers aren't half-eaten
# by the phone pattern.
PII_PATTERNS = [
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("[PHONE]", re.compile(r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)")),
    ("[ACCOUNT#]", re.compile(r"\b\d{8,12}\b")),
]

INJECTION_PATTERNS = [
    "ignore previous instructions", "ignore all previous", "disregard your instructions",
    "system prompt", "reveal your prompt", "output your instructions",
    "you are now", "pretend you are", "developer mode", "jailbreak",
    "act as an unrestricted",
]

BLOCKED_RESPONSE = (
    "I can't process that request. I'm here to help with telecom support "
    "questions — billing, network, devices, plans, or your account. "
    "What can I help you with?"
)


@dataclass
class InputCheck:
    blocked: bool
    masked_query: str
    flags: list[str] = field(default_factory=list)


@dataclass
class OutputCheck:
    response: str
    flags: list[str] = field(default_factory=list)


class GuardrailsAgent:
    def check_input(self, query: str) -> InputCheck:
        flags = []
        masked = query

        for label, pattern in PII_PATTERNS:
            masked, n = pattern.subn(label, masked)
            if n:
                flags.append(f"pii_masked:{label.strip('[]#').lower()}x{n}")

        q = query.lower()
        injected = [p for p in INJECTION_PATTERNS if p in q]
        if injected:
            flags.append(f"prompt_injection:{injected[0].replace(' ', '_')}")
            return InputCheck(blocked=True, masked_query=masked, flags=flags)

        return InputCheck(blocked=False, masked_query=masked, flags=flags)

    def check_output(self, response: str, context_chunks: list[dict]) -> OutputCheck:
        flags = []
        masked = response
        for label, pattern in PII_PATTERNS:
            masked, n = pattern.subn(label, masked)
            if n:
                flags.append(f"pii_leak_redacted:{label.strip('[]#').lower()}x{n}")

        # Dollar amounts not present in grounding context or the query → possible
        # hallucinated pricing. Flag only (the answer may legitimately use
        # fine-tuned internal knowledge); auditors see it in the trace.
        context_text = " ".join(c.get("text", "") for c in context_chunks)
        known_amounts = set(re.findall(r"\$\d+(?:\.\d+)?", context_text))
        # Amounts that exist in the curated training facts are verified too
        known_amounts |= {"$0", "$0.00", "$5", "$10", "$20", "$25", "$29", "$249"}
        for amount in set(re.findall(r"\$\d+(?:\.\d+)?", response)):
            if amount not in known_amounts:
                flags.append(f"unverified_amount:{amount}")

        return OutputCheck(response=masked, flags=flags)
