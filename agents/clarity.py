"""Clarity Assurance Agent — ambiguity gate before the pipeline.

LLMs answer first and never ask. This agent judges whether a query carries
enough signal to answer accurately; if not, it returns a targeted clarifying
question instead of letting the model guess (hallucination level one).

Deterministic heuristics — fast, predictable, auditable. With conversation
history present, the gate relaxes: elliptical follow-ups are expected.
"""

from __future__ import annotations
import re
from dataclasses import dataclass

# Strong signals the query is answerable as-is
CODE_PATTERN = re.compile(r"\b[A-Z]{1,4}-\d{2,4}[A-Za-z]*\b")
DOMAIN_KEYWORDS = {
    "bill", "billing", "charge", "payment", "invoice", "fee", "refund", "autopay",
    "signal", "coverage", "internet", "data", "5g", "wifi", "roaming", "network",
    "phone", "device", "router", "modem", "gateway", "sim", "esim", "unlock",
    "plan", "upgrade", "unlimited", "international", "account", "password",
    "port", "cancel", "voicemail", "hotspot", "fiber", "speed", "outage",
}
VAGUE_ONLY = {"not working", "doesn't work", "is broken", "have a problem",
              "have an issue", "need help", "something wrong", "not good"}


@dataclass
class ClarityResult:
    clear: bool
    clarifying_question: str | None = None
    reason: str = "clear"


class ClarityAgent:
    def check(self, query: str, history: str | None = None) -> ClarityResult:
        q = query.strip()
        ql = q.lower()
        words = ql.split()

        # Internal codes/hardware are maximally specific
        if CODE_PATTERN.search(q):
            return ClarityResult(clear=True, reason="contains specific code/model")

        # Follow-ups lean on memory; don't interrogate the customer twice
        if history:
            return ClarityResult(clear=True, reason="follow-up with conversation context")

        has_domain = any(k in DOMAIN_KEYWORDS for k in words)
        is_vague = any(p in ql for p in VAGUE_ONLY)

        if len(words) < 3 and not has_domain:
            return ClarityResult(
                clear=False, reason="too short, no domain signal",
                clarifying_question=(
                    "I'd love to help — could you tell me a bit more? "
                    "Is this about your bill, network/internet, a device, "
                    "your plan, or your account?"))

        if is_vague and not has_domain:
            return ClarityResult(
                clear=False, reason="vague problem statement, no subject",
                clarifying_question=(
                    "Sorry you're having trouble! To point you in the right "
                    "direction: what exactly isn't working — your mobile data, "
                    "calls, home internet, a device, or something on your bill?"))

        if not has_domain and len(words) < 6:
            return ClarityResult(
                clear=False, reason="off-domain or underspecified",
                clarifying_question=(
                    "Could you give me a little more detail about the issue? "
                    "For example the service affected (mobile, internet, billing) "
                    "and what you're seeing."))

        return ClarityResult(clear=True, reason="sufficient detail")
