from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvironmentInfo:
    is_raspberry_pi: bool
    machine: str
    system: str


def _cpuinfo_contains_raspberrypi() -> bool:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return False
    text = cpuinfo.read_text(encoding="utf-8", errors="ignore").lower()
    return "raspberry pi" in text or "bcm" in text


def detect_environment() -> EnvironmentInfo:
    """Auto-detect Raspberry Pi vs regular PC environment."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    is_pi = (
        os.environ.get("FORCE_RPI", "0") == "1"
        or "arm" in machine
        and _cpuinfo_contains_raspberrypi()
    )

    return EnvironmentInfo(
        is_raspberry_pi=is_pi,
        machine=machine,
        system=system,
    )
