"""Telecom enterprise MCP server — internal systems exposed over the
Model Context Protocol.

Tools:
  find_expert(domain)        — on-call expert routing from the (synthetic)
                               employee directory; used when the trust gate
                               escalates an answer to human review
  get_outage_status(area)    — live network status feed (synthetic), injected
                               as grounding evidence for network queries
  get_current_datetime()     — server time, for SLA/billing-cycle context

Run it in a separate terminal (visible proof of a real MCP process):

    python mcp_server/telecom_mcp.py
    # serving on http://0.0.0.0:8765/sse

The TruthLine orchestrator connects as an MCP client at startup; if this
server isn't running, the pipeline degrades gracefully (no expert routing,
no live outage feed) — nothing breaks.
"""

import random
import time
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("telecom-enterprise", host="0.0.0.0", port=8765)


# Synthetic employee directory — production would back this with HR/on-call
# systems (PagerDuty, MS Teams shifts, etc.) behind the same MCP tool.
EXPERTS = [
    {"name": "Priya Sharma",  "role": "CPE & hardware specialist", "domain": "device",
     "on_call": True,  "open_tickets": 3, "contact": "priya.sharma@telco.example"},
    {"name": "Rahul Verma",   "role": "Billing operations lead",   "domain": "billing",
     "on_call": True,  "open_tickets": 5, "contact": "rahul.verma@telco.example"},
    {"name": "Anita Desai",   "role": "Network operations (NOC)",  "domain": "network",
     "on_call": True,  "open_tickets": 2, "contact": "anita.desai@telco.example"},
    {"name": "Vikram Iyer",   "role": "Plans & retention senior",  "domain": "plan",
     "on_call": False, "open_tickets": 0, "contact": "vikram.iyer@telco.example"},
    {"name": "Sara Khan",     "role": "Account security analyst",  "domain": "account",
     "on_call": True,  "open_tickets": 1, "contact": "sara.khan@telco.example"},
    {"name": "Dev Mukherjee", "role": "Customer care supervisor",  "domain": "general",
     "on_call": True,  "open_tickets": 4, "contact": "dev.mukherjee@telco.example"},
]


@mcp.tool()
def find_expert(domain: str) -> dict:
    """Route an escalation to the right on-call domain expert.

    domain: one of billing, network, device, plan, account, general.
    Returns the on-call expert with the lowest open-ticket load for the
    domain, falling back to the customer-care supervisor.
    """
    candidates = [e for e in EXPERTS if e["domain"] == domain and e["on_call"]]
    if not candidates:
        candidates = [e for e in EXPERTS if e["domain"] == "general"]
    expert = min(candidates, key=lambda e: e["open_tickets"])
    return {**expert, "routed_at": datetime.now(timezone.utc).isoformat()}


@mcp.tool()
def get_outage_status(area: str = "customer-area") -> dict:
    """Live network status for an area (synthetic feed; production would hit
    the NOC's status API). Deterministic per 10-minute window so demo runs
    are stable."""
    window = int(time.time() // 600)
    rng = random.Random(f"{area}-{window}")
    outage = rng.random() < 0.25
    return {
        "area": area,
        "outage": outage,
        "class": "Class-2 (regional degradation)" if outage else None,
        "eta_restore": f"{rng.randint(1, 6)}h" if outage else None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
def get_current_datetime() -> str:
    """Current server date/time (UTC) — billing-cycle and SLA context."""
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    print("Telecom enterprise MCP server — http://0.0.0.0:8765/sse")
    print(f"tools: find_expert, get_outage_status, get_current_datetime")
    mcp.run(transport="sse")
