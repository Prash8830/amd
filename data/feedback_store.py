"""Ground-truth feedback store — the data flywheel.

Every thumbs-up in the dashboard appends an approved Q&A pair here. The next
fine-tuning run (finetune.py) automatically merges approved pairs into the
training set — user feedback literally becomes model weights. On the MI300X a
retrain costs ~1 minute, so the flywheel can spin nightly (or hourly).

Thumbs-down pairs are stored too: they're excluded from training and form a
review queue for the knowledge team.
"""

from __future__ import annotations
import json
import os
import time

FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "feedback/ground_truth.jsonl")


def append_feedback(question: str, answer: str, label: str) -> None:
    """label: 'approved' (thumbs up) or 'rejected' (thumbs down)."""
    os.makedirs(os.path.dirname(FEEDBACK_PATH) or ".", exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps({
            "question": question,
            "answer": answer,
            "label": label,
            "ts": time.time(),
        }) + "\n")


def load_feedback(label: str | None = None) -> list[dict]:
    if not os.path.exists(FEEDBACK_PATH):
        return []
    rows = []
    with open(FEEDBACK_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if label is None or row.get("label") == label:
                rows.append(row)
    return rows


def load_approved() -> list[dict]:
    """Deduplicated approved pairs, latest label wins (a later thumbs-down
    on the same question/answer revokes an earlier approval)."""
    latest: dict[tuple, str] = {}
    for row in load_feedback():
        latest[(row["question"], row["answer"])] = row.get("label", "")
    return [{"question": q, "answer": a}
            for (q, a), label in latest.items() if label == "approved"]
