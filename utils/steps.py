"""Step visualizer — prints clear progress banners with timing, flushed live to notebooks."""

from __future__ import annotations
import sys
import time


class StepTracker:
    """Prints numbered step banners with elapsed time.

    Usage:
        steps = StepTracker(total=5)
        with steps.step("Loading model"):
            ...
    """

    def __init__(self, total: int, title: str = ""):
        self.total = total
        self.current = 0
        self.t_start = time.perf_counter()
        if title:
            bar = "=" * 62
            print(f"\n{bar}\n  {title}\n{bar}", flush=True)

    def step(self, description: str) -> "_Step":
        self.current += 1
        return _Step(self, description)

    def done(self):
        total_s = time.perf_counter() - self.t_start
        print(f"\n{'=' * 62}\n  ALL DONE in {_fmt(total_s)}\n{'=' * 62}", flush=True)


class _Step:
    def __init__(self, tracker: StepTracker, description: str):
        self.tracker = tracker
        self.description = description

    def __enter__(self):
        self.t0 = time.perf_counter()
        n, total = self.tracker.current, self.tracker.total
        print(f"\n[STEP {n}/{total}] {self.description} ...", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.t0
        n, total = self.tracker.current, self.tracker.total
        if exc_type is None:
            print(f"[STEP {n}/{total}] ✓ {self.description} — {_fmt(elapsed)}", flush=True)
        else:
            print(f"[STEP {n}/{total}] ✗ FAILED after {_fmt(elapsed)}: {exc}", flush=True)
        return False

    def note(self, message: str):
        print(f"    → {message}", flush=True)


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"
