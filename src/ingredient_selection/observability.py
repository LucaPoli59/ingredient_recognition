"""Blinded, deterministic image/ingredient annotation for Phase 3 P5."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PILOT_ID = "phase3-p5-observability-pilot-v1"
PILOT_LABELS = (
    "avocado", "cheese", "chicken broth", "flour", "vanilla extract",
    "salt", "soy sauce", "tomato",
)
CATEGORIES = ("direct", "contextual", "not_inferable", "uncertain")
POSITIVES_PER_LABEL = 6
NEGATIVES_PER_LABEL = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"existing review artifact differs: {path}")
        return
    path.write_bytes(content)


def _load_basis(repo_root: Path, campaign_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((campaign_dir / "campaign_manifest.json").read_text())
    validation = json.loads((campaign_dir / "validation_summary.json").read_text())
    report = json.loads((campaign_dir / "p4_profile_report.json").read_text())
    rule = json.loads((campaign_dir / "profile_rule.json").read_text())
    evidence_path = campaign_dir / "profile_evidence.csv"
    if (manifest.get("status") != "completed"
            or validation.get("analysis_scope") != "full"
            or validation.get("test_split_accessed") is not False
            or validation.get("campaign_identity_hash") != manifest.get("campaign_identity_hash")
            or report.get("campaign_identity_hash") != manifest.get("campaign_identity_hash")
            or report.get("profile_rule_hash") != rule.get("artifact_hash")
            or report.get("profile_evidence_sha256") != _sha256(evidence_path)
            or validation.get("profile_evidence_sha256") != _sha256(evidence_path)):
        raise ValueError("P5 requires the completed, unchanged and test-isolated P4 evidence")
    class_order = manifest["campaign_identity"]["class_order"]
    with evidence_path.open(newline="", encoding="utf-8") as stream:
        evidence = list(csv.DictReader(stream))
    if (len(evidence) != len(class_order)
            or [row["class_name"] for row in evidence] != class_order
            or [int(row["class_index"]) for row in evidence] != list(range(len(class_order)))):
        raise ValueError("P4 evidence does not match the campaign class order")
    for name in PILOT_LABELS:
        if name not in class_order:
            raise ValueError(f"pilot label missing from campaign: {name}")
    val_path = repo_root / "data/input/yummly/val/ingredients_target_v5_metadata.json"
    expected = manifest["campaign_identity"]["metadata_sha256"]["val"]
    if _sha256(val_path) != expected:
        raise ValueError("validation metadata differs from the campaign")
    records = json.loads(val_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len({str(r["id"]) for r in records}) != len(records):
        raise ValueError("validation metadata is empty or has duplicate record IDs")
    return manifest, evidence, records


def _rank(label: str, record_id: str) -> str:
    return hashlib.sha256(f"{PILOT_ID}\0{label}\0{record_id}".encode()).hexdigest()


def _sample_pairs(labels: tuple[str, ...], records: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    used_images: set[str] = set()
    for label in labels:
        for recipe_label_present, count in ((True, POSITIVES_PER_LABEL),
                                            (False, NEGATIVES_PER_LABEL)):
            eligible = [row for row in records
                        if (label in row["ingredients_target"]) is recipe_label_present]
            eligible.sort(key=lambda row: (_rank(label, str(row["id"])), str(row["id"])))
            chosen = []
            for row in eligible:
                if row["image"] not in used_images:
                    chosen.append(row)
                    used_images.add(row["image"])
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f"insufficient distinct validation images for {label}")
            for row in chosen:
                record_id = str(row["id"])
                pair_id = hashlib.sha256(
                    f"{PILOT_ID}\0{label}\0{record_id}".encode()).hexdigest()[:20]
                pairs.append({
                    "pair_id": pair_id, "ingredient": label, "record_id": record_id,
                    "image": str(row["image"]), "recipe_label_present": recipe_label_present,
                })
    if len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("duplicate pair ID")
    return pairs


def _reviewer_html(rows: list[dict], reviewer: str, packet_hash: str) -> bytes:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    storage = f"{PILOT_ID}:{packet_hash}:{reviewer}"
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Ingredient observability review</title>
<style>body{font:17px system-ui;max-width:860px;margin:2rem auto;padding:0 1rem}
img{max-width:100%;max-height:560px;object-fit:contain;background:#eee}
label{display:block;margin:.8rem 0}button{margin:.6rem .6rem .6rem 0;padding:.45rem .7rem}
textarea{width:100%;height:4rem}small{color:#555}</style>
<h1>Ingredient observability review</h1>
<p>Judge this ingredient from the photograph alone. Do not consult the recipe,
model scores, or another reviewer. <b>Direct</b>: identifiable visual evidence
of the ingredient itself. <b>Contextual</b>: plausible from dish appearance but
not individually identifiable. <b>Not inferable</b>: no meaningful image cue.
<b>Uncertain</b>: ambiguous image/ingredient or insufficient quality. A recipe
ingredient need not be visible; do not guess the hidden recipe label.</p>
<p><small>Use a different reviewer identity/person for each packet. Browser state
is local; export a JSON backup before closing this page.</small></p>
<label>Reviewer name or pseudonym <input id="name" autocomplete="off"></label>
<p id="progress"></p><h2 id="ingredient"></h2><img id="photo" alt="Food photograph">
<div id="choices"></div><label>Optional reason for uncertainty or ambiguity
<textarea id="note"></textarea></label>
<button id="prev">Previous</button><button id="next">Next</button>
<button id="export">Export JSON</button>
<script>
const rows=__ROWS__, reviewer=__REVIEWER__, packetHash=__HASH__, key=__KEY__;
const cats=['direct','contextual','not_inferable','uncertain'];
let saved=JSON.parse(localStorage.getItem(key)||'{"name":"","answers":{}}'), pos=0;
const $=id=>document.getElementById(id);
function persist(){saved.name=$('name').value.trim();localStorage.setItem(key,JSON.stringify(saved));}
function render(){const row=rows[pos], answer=saved.answers[row.pair_id]||{};
 $('progress').textContent=`${pos+1}/${rows.length} · ${Object.keys(saved.answers).length} answered`;
 $('ingredient').textContent=`Could you infer: ${row.ingredient}?`;
 $('photo').src=row.image_href; $('choices').replaceChildren();
 for(const cat of cats){const label=document.createElement('label');
  const radio=document.createElement('input');radio.type='radio';radio.name='category';
  radio.value=cat;radio.checked=answer.category===cat;
  radio.onchange=()=>{saved.answers[row.pair_id]={category:cat,note:$('note').value.trim()};persist();render();};
  label.append(radio,` ${cat.replaceAll('_',' ')}`);$('choices').append(label);}
 $('note').value=answer.note||'';
}
$('name').value=saved.name;$('name').oninput=persist;
$('note').oninput=()=>{const row=rows[pos];if(saved.answers[row.pair_id]){
 saved.answers[row.pair_id].note=$('note').value.trim();persist();}};
$('prev').onclick=()=>{pos=Math.max(0,pos-1);render();};
$('next').onclick=()=>{pos=Math.min(rows.length-1,pos+1);render();};
$('export').onclick=()=>{persist();if(!saved.name){alert('Enter a reviewer name.');return;}
 const result={schema_version:1,reviewer_id:reviewer,reviewer_name:saved.name,
  packet_hash:packetHash,responses:saved.answers};
 const blob=new Blob([JSON.stringify(result,null,2)+'\\n'],{type:'application/json'});
 const link=document.createElement('a');link.href=URL.createObjectURL(blob);
 link.download=`p5_pilot_review_${reviewer}.json`;link.click();URL.revokeObjectURL(link.href);};
render();</script></html>"""
    return (page.replace("__ROWS__", data).replace("__REVIEWER__", json.dumps(reviewer))
            .replace("__HASH__", json.dumps(packet_hash))
            .replace("__KEY__", json.dumps(storage))).encode("utf-8")


def prepare_pilot(repo_root: Path, campaign_dir: Path, output_dir: Path) -> dict:
    repo_root = repo_root.resolve()
    campaign_dir = campaign_dir.resolve()
    output_dir = output_dir.resolve()
    manifest, evidence, records = _load_basis(repo_root, campaign_dir)
    pairs = _sample_pairs(PILOT_LABELS, records)
    image_root = (repo_root / "data/input/yummly/imgs/standard").resolve()
    for pair in pairs:
        image = (image_root / pair["image"]).resolve()
        if not image.is_file() or not image.is_relative_to(image_root):
            raise ValueError(f"missing or invalid image: {pair['image']}")
        pair["image_sha256"] = _sha256(image)
    source = {
        "schema_version": 1,
        "packet_id": PILOT_ID,
        "campaign_identity_hash": manifest["campaign_identity_hash"],
        "profile_evidence_sha256": _sha256(campaign_dir / "profile_evidence.csv"),
        "profile_rule_hash": json.loads((campaign_dir / "profile_rule.json").read_text())["artifact_hash"],
        "val_metadata_sha256": manifest["campaign_identity"]["metadata_sha256"]["val"],
        "protocol_source_sha256": _sha256(Path(__file__)),
        "pilot_labels": list(PILOT_LABELS),
        "profile_outcomes_for_stratification": {
            row["class_name"]: row["provisional_outcome"] for row in evidence
            if row["class_name"] in PILOT_LABELS
        },
        "positive_pairs_per_label": POSITIVES_PER_LABEL,
        "negative_pairs_per_label": NEGATIVES_PER_LABEL,
        "pairs": pairs,
    }
    source["packet_hash"] = hashlib.sha256(_json_bytes(source)).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_once(output_dir / "packet_manifest.json", _json_bytes(source))
    for reviewer in ("A", "B"):
        ordered = sorted(pairs, key=lambda pair: hashlib.sha256(
            f"{source['packet_hash']}\0{reviewer}\0{pair['pair_id']}".encode()).hexdigest())
        public = [{"pair_id": pair["pair_id"], "ingredient": pair["ingredient"],
                   "image_href": os.path.relpath(
                       image_root / pair["image"], output_dir).replace(os.sep, "/")}
                  for pair in ordered]
        _write_once(output_dir / f"reviewer_{reviewer.lower()}.html",
                    _reviewer_html(public, reviewer, source["packet_hash"]))
    return source


def _validated_responses(path: Path, reviewer: str, packet: dict) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = {row["pair_id"] for row in packet["pairs"]}
    if (payload.get("schema_version") != 1
            or payload.get("reviewer_id") != reviewer
            or payload.get("packet_hash") != packet["packet_hash"]
            or not isinstance(payload.get("reviewer_name"), str)
            or not payload["reviewer_name"].strip()
            or not isinstance(payload.get("responses"), dict)
            or set(payload["responses"]) != expected_ids):
        raise ValueError(f"incomplete or mismatched reviewer {reviewer} response")
    responses = payload["responses"]
    if any(not isinstance(answer, dict) or answer.get("category") not in CATEGORIES
           for answer in responses.values()):
        raise ValueError(f"invalid reviewer {reviewer} category")
    return {key: answer["category"] for key, answer in responses.items()}


def score_pilot(packet_path: Path, reviewer_a: Path, reviewer_b: Path,
                image_root: Path | None = None) -> dict:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_hash = packet.pop("packet_hash")
    if hashlib.sha256(_json_bytes(packet)).hexdigest() != packet_hash:
        raise ValueError("packet manifest was modified")
    packet["packet_hash"] = packet_hash
    if image_root is not None:
        image_root = image_root.resolve()
        for pair in packet["pairs"]:
            image = (image_root / pair["image"]).resolve()
            if (not image.is_relative_to(image_root) or not image.is_file()
                    or _sha256(image) != pair["image_sha256"]):
                raise ValueError(f"review image changed or missing: {pair['pair_id']}")
    names = [json.loads(p.read_text(encoding="utf-8"))["reviewer_name"].strip().casefold()
             for p in (reviewer_a, reviewer_b)]
    if len(set(names)) != 2:
        raise ValueError("two distinct reviewer identifiers are required")
    a = _validated_responses(reviewer_a, "A", packet)
    b = _validated_responses(reviewer_b, "B", packet)
    matrix = {left: {right: 0 for right in CATEGORIES} for left in CATEGORIES}
    per_label: dict[str, dict[str, Any]] = {}
    for pair in packet["pairs"]:
        left, right = a[pair["pair_id"]], b[pair["pair_id"]]
        matrix[left][right] += 1
        if pair["recipe_label_present"]:
            row = per_label.setdefault(pair["ingredient"], {
                "positive_recipe_pairs": 0, "agreed_direct": 0,
                "agreed_contextual": 0, "agreed_not_inferable": 0,
                "agreed_uncertain": 0, "disagreements": 0,
            })
            row["positive_recipe_pairs"] += 1
            if left == right:
                row[f"agreed_{left}"] += 1
            else:
                row["disagreements"] += 1
    n = len(packet["pairs"])
    observed = sum(matrix[c][c] for c in CATEGORIES) / n
    chance = sum(
        sum(matrix[c].values()) * sum(matrix[r][c] for r in CATEGORIES)
        for c in CATEGORIES
    ) / (n * n)
    return {
        "schema_version": 1, "packet_id": PILOT_ID,
        "packet_hash": packet_hash, "reviewer_response_sha256": {
            "A": _sha256(reviewer_a), "B": _sha256(reviewer_b)},
        "pairs": n, "observed_agreement": observed,
        "cohen_kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
        "confusion_matrix": matrix, "per_label_positive_recipe_pairs": per_label,
        "interpretation": "Pilot reliability only; no adjudication, tier, or final vocabulary",
    }
