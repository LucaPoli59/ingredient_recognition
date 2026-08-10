"""Build the pinned compact FoodOn index used by the Yummly target generator."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.foodon_lexicon import load_foodon_tsv, write_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_tsv", type=Path, help="Pinned FoodOn root foodon-synonyms.tsv export")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/data_processing/resources/foodon_food_product_v2025_07_31.json"),
    )
    parser.add_argument("--release", default="FoodOn v2025-07-31")
    parser.add_argument("--commit", default="7ede44c")
    args = parser.parse_args()

    digest = hashlib.sha256(args.source_tsv.read_bytes()).hexdigest()
    lexicon, provenance = load_foodon_tsv(args.source_tsv)
    provenance = {
        **provenance,
        "source_release": args.release,
        "source_commit": args.commit,
        "source_sha256": digest,
    }
    write_index(lexicon, args.output, provenance)
    print(f"wrote {args.output} ({lexicon.concept_count} concepts, {len(lexicon.surface_to_ids)} surfaces)")
    print(f"source sha256: {digest}")


if __name__ == "__main__":
    main()
