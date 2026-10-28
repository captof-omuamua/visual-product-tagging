#!/usr/bin/env python3
"""Check this machine can run stage 1 before anything downloads (standard library only).

Writes reports/env_check.json, prints what to fix, and exits 1 if a blocker is found.

Blockers: not Linux, no NVIDIA driver, a driver too old for the vLLM wheels, no uv,
or under 80 GB free. Warnings: less than 20 GB of VRAM, or under 110 GB free.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# vLLM 0.32.0's PyPI wheels are built with CUDA 13.0, so the driver must support 13.0+.
MIN_DRIVER_CUDA = (13, 0)
MIN_FREE_GB = 80
WARN_FREE_GB = 110
MIN_VRAM_GB = 20


def run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        return out.stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def parse_cuda_version(nvidia_smi_text: str | None) -> tuple[int, int] | None:
    """The highest CUDA version the driver supports, from nvidia-smi's banner."""
    if not nvidia_smi_text:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", nvidia_smi_text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_gpu_query(csv_text: str | None) -> list[dict]:
    """Rows of `nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap`."""
    gpus = []
    for line in (csv_text or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, driver, memory, cap = parts[:4]
        mib = re.search(r"(\d+)", memory)
        gpus.append(
            {
                "name": name,
                "driver": driver,
                "vram_gb": round(int(mib.group(1)) / 1024, 1) if mib else None,
                "compute_capability": cap,
            }
        )
    return gpus


def free_gb(path: Path) -> float:
    path = path if path.exists() else path.parent
    while not path.exists():
        path = path.parent
    return round(shutil.disk_usage(path).free / 1024**3, 1)


def evaluate(
    system: str,
    gpus: list[dict],
    driver_cuda: tuple[int, int] | None,
    has_uv: bool,
    disk_free_gb: float,
) -> tuple[list[str], list[str]]:
    """Turn facts into (blockers, warnings), each with the fix spelled out."""
    blockers: list[str] = []
    warnings: list[str] = []
    if system != "Linux":
        blockers.append(
            f"This is {system}. Run stage 1 in your Ubuntu install: vLLM needs Linux."
        )
    if not gpus:
        blockers.append(
            "No NVIDIA GPU found (nvidia-smi failed). Install the driver with "
            "`sudo ubuntu-drivers install`, reboot, then rerun."
        )
    if gpus and driver_cuda is None:
        warnings.append("Could not read the driver's CUDA version from nvidia-smi.")
    if driver_cuda is not None and driver_cuda < MIN_DRIVER_CUDA:
        need = ".".join(map(str, MIN_DRIVER_CUDA))
        have = ".".join(map(str, driver_cuda))
        blockers.append(
            f"Your driver supports CUDA {have}; vLLM 0.32's wheels need CUDA {need}+. "
            "Update the driver (`sudo ubuntu-drivers install`, or "
            "`sudo apt install nvidia-driver-580-open` or newer), reboot, rerun. RTX 50 "
            "GPUs need the open kernel modules (the -open packages)."
        )
    for gpu in gpus:
        if gpu["vram_gb"] is not None and gpu["vram_gb"] < MIN_VRAM_GB:
            warnings.append(
                f"{gpu['name']} has {gpu['vram_gb']} GB; stage 1 assumes about 24 GB."
            )
    if not has_uv:
        blockers.append(
            "uv is missing. Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` "
            "(or `wget -qO- https://astral.sh/uv/install.sh | sh`), open a new terminal, rerun."
        )
    if disk_free_gb < MIN_FREE_GB:
        blockers.append(
            f"Only {disk_free_gb} GB free; stage 1 needs about {MIN_FREE_GB}-{WARN_FREE_GB} GB "
            "(three Python environments, Qwen3.5-4B and its merged copy, two ~18 GB large "
            "models and the ABO listings)."
        )
    elif disk_free_gb < WARN_FREE_GB:
        warnings.append(f"{disk_free_gb} GB free; {WARN_FREE_GB} GB leaves room for stage 2 data.")
    return blockers, warnings


def main() -> int:
    banner = run(["nvidia-smi"])
    query = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    gpus = parse_gpu_query(query)
    driver_cuda = parse_cuda_version(banner)
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    disk = min(free_gb(ROOT), free_gb(hf_home))
    has_uv = shutil.which("uv") is not None
    blockers, warnings = evaluate(platform.system(), gpus, driver_cuda, has_uv, disk)

    report = {
        "ok": not blockers,
        "system": platform.system(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "gpus": gpus,
        "driver_cuda": ".".join(map(str, driver_cuda)) if driver_cuda else None,
        "uv": shutil.which("uv"),
        "disk_free_gb": disk,
        "blockers": blockers,
        "warnings": warnings,
    }
    out = ROOT / "reports" / "env_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for gpu in gpus:
        print(
            f"GPU: {gpu['name']}, {gpu['vram_gb']} GB, driver {gpu['driver']}, "
            f"compute capability {gpu['compute_capability']}"
        )
    print(f"Driver CUDA: {report['driver_cuda']}; free disk: {disk} GB; uv: {report['uv']}")
    for w in warnings:
        print(f"WARN: {w}")
    for b in blockers:
        print(f"BLOCKER: {b}")
    print("Environment check passed." if not blockers else "Fix the blockers above, then rerun.")
    return 0 if not blockers else 1


if __name__ == "__main__":
    sys.exit(main())
