"""Fresh frozen DINOv2-B/14 probing on D6 with transferred trial-61 settings."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "settings"):
    sys.path.insert(0, str(path))

from src.training.selected_vocab import main


if __name__ == "__main__":
    main("dinov2")
