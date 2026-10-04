"""Prepare and score the Phase 3 P5 two-reviewer observability pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingredient_selection.observability import (  # noqa: E402
    _json_bytes,
    _write_once,
    prepare_pilot,
    score_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--campaign", type=Path, default=ROOT / "analysis_outputs/ingredient_selection/phase3-d1-v3")
    prepare.add_argument("--output", type=Path, default=ROOT / "analysis_outputs/ingredient_selection/phase3-d1-v3/p5_observability/pilot")
    score = sub.add_parser("score")
    score.add_argument("--packet", type=Path, required=True)
    score.add_argument("--reviewer-a", type=Path, required=True)
    score.add_argument("--reviewer-b", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        packet = prepare_pilot(ROOT, args.campaign, args.output)
        print(f"prepared {len(packet['pairs'])} blind pairs: {args.output}")
    else:
        result = score_pilot(args.packet, args.reviewer_a, args.reviewer_b,
                             ROOT / "data/input/yummly/imgs/standard")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_once(args.output, _json_bytes(result))
        print(f"scored {result['pairs']} pairs; agreement={result['observed_agreement']:.3f}")


if __name__ == "__main__":
    main()
