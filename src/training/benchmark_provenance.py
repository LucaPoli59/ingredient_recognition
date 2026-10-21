"""Explicit, additive provenance for the reviewed final-checkpoint-only repair."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
from functools import partial

from src.training.benchmark_protocol import policy as load_policy, policy_hash as hash_policy, LEGACY_PROTOCOL_ID

# This historical patch never follows the active new-study policy implicitly.
policy = partial(load_policy, LEGACY_PROTOCOL_ID)
policy_hash = partial(hash_policy, LEGACY_PROTOCOL_ID)

PATCH_ID = "final-checkpoint-v1"
MODIFIED = {"src/lightning/benchmark_lgn.py", "src/training/benchmark_campaign.py"}
ADDED = {"src/training/benchmark_provenance.py", "scripts/launch_exps/benchmark/register_runtime_patch.py"}
REPO = Path(__file__).resolve().parents[2]


def sha256(path, *, length=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        remaining = length
        while remaining is None or remaining > 0:
            chunk = stream.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not chunk:
                if remaining:
                    raise ValueError("preserved journal prefix was truncated")
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def source_files(repo=REPO):
    repo = Path(repo)
    paths = set(p for directory in ("src", "settings", "scripts/launch_exps/benchmark")
                for p in (repo / directory).rglob("*.py"))
    paths.update((repo / "src/training/resources").glob("*.json"))
    return {str(path.relative_to(repo)): sha256(path) for path in sorted(paths)}


def write_source_snapshot(destination, repo=REPO):
    destination, repo = Path(destination), Path(repo)
    files = source_files(repo)
    with zipfile.ZipFile(destination / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(repo / name, name)
    inventory = {"base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                 "files": files, "zip_sha256": sha256(destination / "source_snapshot.zip")}
    write_new(destination / "source_inventory.json", inventory)
    return inventory


def write_new(path, values):
    with Path(path).open("x") as stream:
        json.dump(values, stream, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_snapshot(directory):
    directory = Path(directory)
    saved = read_json(directory / "source_inventory.json")
    if sha256(directory / "source_snapshot.zip") != saved["zip_sha256"]:
        raise ValueError("saved source ZIP differs")
    with zipfile.ZipFile(directory / "source_snapshot.zip") as archive:
        if set(archive.namelist()) != set(saved["files"]):
            raise ValueError("saved source inventory/ZIP membership differs")
        if any(hashlib.sha256(archive.read(name)).hexdigest() != digest for name, digest in saved["files"].items()):
            raise ValueError("saved source ZIP entry differs")
    return saved


def patch_directory(root):
    return Path(root) / "runtime_patches" / PATCH_ID


def _check_delta(original, current):
    old, new = set(original), set(current)
    changed = {name for name in old & new if original[name] != current[name]}
    if changed != MODIFIED or new - old != ADDED or old - new:
        raise ValueError("source changes exceed the reviewed final-checkpoint-only repair")


def inspect_patch(root, repo=REPO):
    """Read-only preflight; no journal replay, model construction, or registration."""
    root = Path(root)
    if patch_directory(root).exists():
        return validate_study_sources(root, repo)
    original = verify_snapshot(root)
    _check_delta(original["files"], source_files(repo))
    identity = read_json(root / "study_identity.json")
    if (identity["policy_hash"] != policy_hash() or identity["model_id"] not in policy()["models"]
            or read_json(root / "policy.json") != policy()):
        raise ValueError("scientific study policy differs")
    states, attributes, created = {}, {}, 0
    for line in (root / "journal.log").read_text().splitlines():
        event = json.loads(line)
        if event["op_code"] == 4:
            created += 1
        elif event["op_code"] == 6:
            states[event["trial_id"]] = event["state"]
        elif event["op_code"] == 8:
            attributes.setdefault(event["trial_id"], {}).update(event["user_attr"])
    if (created != 1 or states != {0: 1} or attributes.get(0, {}).get("completed_epochs") != 40
            or attributes.get(0, {}).get("optimizer_updates") != 15000):
        raise ValueError("repair registration requires the single completed first-release trial")
    preserved = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*"))
                 if path.is_file() and "runtime_patches" not in path.relative_to(root).parts
                 and path.name not in ("journal.log", "sampler.bin") and not path.name.endswith(".lock")}
    for required in ("source_inventory.json", "source_snapshot.zip", "trial_0/trial_config.json",
                     "trial_0/epoch_costs.jsonl", "trial_0/checkpoints/last.ckpt",
                     "validation/trial_0/prediction_manifest.json", "validation/trial_0/validation_predictions.npz"):
        if required not in preserved:
            raise ValueError(f"missing retained first-release artifact: {required}")
    sampler = (root / "sampler.bin").read_bytes()
    checksum, payload = sampler.split(b"\n", 1)
    if hashlib.sha256(payload).hexdigest().encode() != checksum:
        raise ValueError("sampler checksum differs")
    return {"patch_id": PATCH_ID, "model_id": identity["model_id"], "policy_hash": policy_hash(),
            "parent_inventory_sha256": sha256(root / "source_inventory.json"),
            "parent_zip_sha256": original["zip_sha256"], "preserved_files": preserved,
            "journal_prefix": {"bytes": (root / "journal.log").stat().st_size, "sha256": sha256(root / "journal.log")},
            "sampler_before_sha256": sha256(root / "sampler.bin"),
            "retained_complete_trials": [0], "missing_final_state_trials": [0]}


def register_checkpoint_patch(root, *, accept_missing_final=False, repo=REPO):
    """Write only a new patch directory; never alter original study artifacts."""
    if not accept_missing_final:
        raise ValueError("explicit acceptance of the retained missing-final exception is required")
    root = Path(root)
    if patch_directory(root).exists():
        return validate_study_sources(root, repo)
    record = inspect_patch(root, repo)
    directory = patch_directory(root)
    directory.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=directory.parent) as temporary:
        temporary = Path(temporary)
        inventory = write_source_snapshot(temporary, repo)
        _check_delta(verify_snapshot(root)["files"], inventory["files"])
        (temporary / "sampler_before.bin").write_bytes((root / "sampler.bin").read_bytes())
        record.update(schema_version=1, registered_at=datetime.now(timezone.utc).isoformat(),
                      active_inventory_sha256=sha256(temporary / "source_inventory.json"),
                      active_zip_sha256=inventory["zip_sha256"],
                      modified_files=sorted(MODIFIED), added_files=sorted(ADDED),
                      reason="Independent every-epoch full checkpoint; explicit immutable source transition. Scientific policy unchanged.",
                      exception_disposition="Keep trial0 COMPLETE and eligible with the missing final state explicitly retained; no replay or retraining.")
        write_new(temporary / "amendment.json", record)
        # Reject concurrent artifact edits before publishing the additive record.
        for name, digest in record["preserved_files"].items():
            if sha256(root / name) != digest:
                raise ValueError("original study artifact changed during registration")
        if (sha256(root / "journal.log") != record["journal_prefix"]["sha256"]
                or sha256(root / "sampler.bin") != record["sampler_before_sha256"]):
            raise ValueError("study advanced during registration")
        temporary.rename(directory)
    return validate_study_sources(root, repo)


def validate_study_sources(root, repo=REPO):
    """Validate original or explicitly amended bytes, including prior evidence."""
    root = Path(root)
    original = verify_snapshot(root)
    current = source_files(repo)
    directory = patch_directory(root)
    if not directory.exists():
        if current != original["files"]:
            raise ValueError("benchmark sources changed; explicit runtime patch registration is required")
        return {"runtime_patch": None, "inventory": "source_inventory.json",
                "inventory_sha256": sha256(root / "source_inventory.json"), "amendment_sha256": None}
    record = read_json(directory / "amendment.json")
    active = verify_snapshot(directory)
    identity = read_json(root / "study_identity.json")
    _check_delta(original["files"], active["files"])
    if (record["schema_version"] != 1 or record["patch_id"] != PATCH_ID
            or record["policy_hash"] != policy_hash() or identity["policy_hash"] != policy_hash()
            or identity["model_id"] != record["model_id"] or read_json(root / "policy.json") != policy()
            or record["parent_inventory_sha256"] != sha256(root / "source_inventory.json")
            or record["parent_zip_sha256"] != original["zip_sha256"]
            or record["active_inventory_sha256"] != sha256(directory / "source_inventory.json")
            or record["active_zip_sha256"] != active["zip_sha256"] or active["files"] != current
            or record["modified_files"] != sorted(MODIFIED) or record["added_files"] != sorted(ADDED)
            or record["retained_complete_trials"] != [0] or record["missing_final_state_trials"] != [0]):
        raise ValueError("registered runtime patch identity/source differs")
    for name, digest in record["preserved_files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(root / name) != digest:
            raise ValueError("preserved first-release artifact differs")
    prefix = record["journal_prefix"]
    if (sha256(root / "journal.log", length=prefix["bytes"]) != prefix["sha256"]
            or sha256(directory / "sampler_before.bin") != record["sampler_before_sha256"]):
        raise ValueError("preserved journal prefix/sampler snapshot differs")
    return {"runtime_patch": PATCH_ID, "inventory": str((directory / "source_inventory.json").relative_to(root)),
            "inventory_sha256": record["active_inventory_sha256"], "amendment_sha256": sha256(directory / "amendment.json")}
