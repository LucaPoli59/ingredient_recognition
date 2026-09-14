"""Write a strict machine-readable report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import json_safe


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(report), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
