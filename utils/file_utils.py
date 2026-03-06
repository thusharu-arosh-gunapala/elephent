from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib


def load_joblib_or_none(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    return joblib.load(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
