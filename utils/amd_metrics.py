"""AMD ROCm GPU metrics collector."""

from __future__ import annotations
import subprocess
import re
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


def get_amd_metrics() -> GPUMetrics:
    """Query AMD GPU stats via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--csv"],
            capture_output=True, text=True, timeout=5
        )
        return _parse_rocm_smi(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: try rocm-smi without --csv
    try:
        result = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=5)
        return _parse_rocm_smi_plain(result.stdout)
    except Exception:
        pass

    return GPUMetrics(
        gpu_utilization_pct=0.0,
        vram_used_mb=0.0,
        vram_total_mb=0.0,
        gpu_temp_c=0.0,
        power_draw_w=0.0,
        available=False,
    )


def _parse_rocm_smi(csv_output: str) -> GPUMetrics:
    lines = [l for l in csv_output.strip().splitlines() if l and not l.startswith("#")]
    if len(lines) < 2:
        return _empty_metrics()

    headers = [h.strip().lower() for h in lines[0].split(",")]
    values = [v.strip() for v in lines[1].split(",")]
    row = dict(zip(headers, values))

    def _f(key: str, default: float = 0.0) -> float:
        for k, v in row.items():
            if key in k:
                try:
                    return float(re.sub(r"[^\d.]", "", v))
                except ValueError:
                    pass
        return default

    return GPUMetrics(
        gpu_utilization_pct=_f("gpu use"),
        vram_used_mb=_f("vram used") / 1024 if _f("vram used") > 1000 else _f("vram used"),
        vram_total_mb=_f("vram total") / 1024 if _f("vram total") > 1000 else _f("vram total"),
        gpu_temp_c=_f("temperature"),
        power_draw_w=_f("power"),
        available=True,
    )


def _parse_rocm_smi_plain(output: str) -> GPUMetrics:
    util = _extract_number(output, r"GPU\s+(\d+)%")
    temp = _extract_number(output, r"(\d+\.?\d*)\s*c")
    vram_used = _extract_number(output, r"(\d+)\s*MB\s*/")
    vram_total = _extract_number(output, r"/\s*(\d+)\s*MB")
    power = _extract_number(output, r"(\d+\.?\d*)\s*W")
    return GPUMetrics(
        gpu_utilization_pct=util,
        vram_used_mb=vram_used,
        vram_total_mb=vram_total,
        gpu_temp_c=temp,
        power_draw_w=power,
        available=True,
    )


def _extract_number(text: str, pattern: str) -> float:
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0


def _empty_metrics() -> GPUMetrics:
    return GPUMetrics(0.0, 0.0, 0.0, 0.0, 0.0, False)


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
