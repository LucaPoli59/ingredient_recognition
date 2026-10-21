"""Read-only verification of registered source repair and unchanged prior trials."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.training.benchmark_campaign import open_study, remaining_trials, select_trial
from src.training.benchmark_provenance import sha256, validate_study_sources, read_json


def verify():
    original = read_json(REPO / "src_scratches/phase6_first_release_audit/summary.json")
    for name, digest in original["input_sha256"].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"original reviewed input changed: {name}")
    models = []
    for row in original["models"]:
        root = REPO / original["campaign"] / row["model_id"]
        sources = validate_study_sources(root)
        before = {name: sha256(root / name) for name in ("journal.log", "sampler.bin")}
        study = open_study(root, row["model_id"], resume=True)
        selected = select_trial(study)
        if (len(study.trials) != 1 or remaining_trials(study) != 99 or selected.number != 0
                or selected.user_attrs["completed_epochs"] != 40
                or selected.user_attrs["optimizer_updates"] != 15000):
            raise ValueError("retained trial identity/accounting changed")
        if any(sha256(root / name) != digest for name, digest in before.items()):
            raise ValueError("read-only study load changed journal/sampler")
        inventory = read_json(root / sources["inventory"])
        models.append({"model_id": row["model_id"], "trials_started": 1, "remaining": 99,
                       "source_identity": sources, "source_entries": len(inventory["files"]),
                       "source_zip_sha256": inventory["zip_sha256"],
                       "journal_sha256": before["journal.log"], "sampler_sha256": before["sampler.bin"]})
    return {"campaign": original["campaign"], "policy_hash": original["policy_hash"],
            "original_review_sha256": sha256(REPO / "src_scratches/phase6_first_release_audit/summary.json"),
            "original_inputs_verified": len(original["input_sha256"]), "models": models,
            "training_started": False, "new_trials_created": False, "test_metadata_or_images_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2))
