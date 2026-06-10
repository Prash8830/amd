"""AMD ROCm GPU metrics collector — rocm-smi JSON output."""

from __future__ import annotations
import json
import re
import subprocess
import psutil
from dataclasses import dataclass


@dataclass
class GPUMetrics:
    gpu_utilization_pct: float
    vram_used_mb: float
    vram_total_mb: float
    gpu_temp_c: float
    power_draw_w: float
    available: bool


def _num(value) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", str(value)) or 0)
    except ValueError:
        return 0.0


def _to_mb(value: float) -> float:
    """rocm-smi reports VRAM in bytes on recent versions, MB on older ones."""
    return value / (1024 ** 2) if value > 1e9 else value


def get_amd_metrics() -> GPUMetrics:
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram",
             "--showtemp", "--showpower", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        data = json.loads(out.stdout)
        card = next(iter(data.values()))  # first GPU, e.g. data["card0"]
    except Exception:
        return GPUMetrics(0.0, 0.0, 0.0, 0.0, 0.0, available=False)

    util = temp = power = vram_used = vram_total = 0.0
    for key, value in card.items():
        kl = key.lower()
        if "gpu use" in kl:
            util = _num(value)
        elif "vram" in kl and "used" in kl:
            vram_used = _num(value)
        elif "vram" in kl and "total" in kl:
            vram_total = _num(value)
        elif "temperature" in kl and ("edge" in kl or "junction" in kl) and temp == 0.0:
            temp = _num(value)
        elif "power" in kl and power == 0.0:
            power = _num(value)

    return GPUMetrics(
        gpu_utilization_pct=util,
        vram_used_mb=_to_mb(vram_used),
        vram_total_mb=_to_mb(vram_total),
        gpu_temp_c=temp,
        power_draw_w=power,
        available=True,
    )


def get_system_metrics() -> dict:
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    gpu = get_amd_metrics()
    return {
        "cpu_pct": cpu,
        "ram_used_gb": round(ram.used / 1e9, 2),
        "ram_total_gb": round(ram.total / 1e9, 2),
        "gpu": gpu,
    }
