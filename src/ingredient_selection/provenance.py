"""Content-addressed source snapshots for runs in the main working tree."""

from pathlib import Path
import subprocess
import zipfile

from src.ingredient_selection.protocol import sha256_file, sha256_json


def source_identity(root: Path) -> dict:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True,
                             capture_output=True).stdout.decode().split("\0")
    paths = {path for path in tracked if path.endswith(".py")}
    for directory in ("src/ingredient_selection", "scripts/ingredient_selection",
                      "scripts/launch_exps/ingredient_selection"):
        paths.update(str(path.relative_to(root)) for path in (root / directory).rglob("*.py"))
    files = {path: sha256_file(root / path) for path in sorted(paths) if (root / path).is_file()}
    return {"files": files, "sha256": sha256_json(files)}


def snapshot_sources(root: Path, destination: Path, identity: dict) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, expected_hash in identity["files"].items():
            if sha256_file(root / path) != expected_hash:
                raise RuntimeError(f"source changed before snapshot: {path}")
            archive.write(root / path, path)
    return {"path": str(destination.relative_to(root)), "sha256": sha256_file(destination)}
