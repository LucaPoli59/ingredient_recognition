"""Export one real parameter-histogram trajectory as a feasibility example.

This reads the audited ResNet trial_72 session only, using the W&B 0.28.0
internal file format. It does not reconstruct individual parameter values.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

from audit import ROOT, rel

OUT = Path(__file__).resolve().parent / "parameter_histogram_example.json"
SUFFIX = "model.layer4.1.conv2.weight"


def sha256(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    inventory = json.loads((OUT.parent / "inventory.json").read_text())
    source = ROOT / next(s["path"] for s in inventory["wandb_samples"]
                         if "resnets_htuning" in s["path"])
    before = sha256(source)
    ds = DataStore()
    samples = []
    keys = set()
    # Read-only equivalent of open_for_scan in the inspected SDK.
    ds._fname = str(source)
    ds._index = 0
    ds._size_bytes = source.stat().st_size
    ds._opened_for_scan = True
    with source.open("rb") as ds._fp:
        ds._read_header()
        while (raw := ds.scan_data()) is not None:
            record = pb.Record()
            record.ParseFromString(raw)
            if record.WhichOneof("record_type") != "history":
                continue
            clocks, histograms = {}, defaultdict(dict)
            for item in record.history.item:
                parts = list(item.nested_key) or [item.key]
                if len(parts) == 1 and parts[0] in (
                        "epoch", "trainer/global_step", "_step", "_timestamp", "_runtime"):
                    clocks[parts[0]] = json.loads(item.value_json)
                if parts[0].startswith("parameters/") and parts[0].endswith(SUFFIX):
                    if len(parts) == 2:
                        histograms[parts[0]][parts[1]] = json.loads(item.value_json)
                    elif len(parts) == 1:
                        histograms[parts[0]] = json.loads(item.value_json)
            for key, hist in histograms.items():
                if hist.get("_type") != "histogram":
                    raise ValueError(f"Unexpected histogram type for {key}")
                edges, counts = hist["bins"], hist["values"]
                assert len(edges) == len(counts) + 1
                assert all(math.isfinite(v) for v in edges + counts)
                assert all(a <= b for a, b in zip(edges, edges[1:]))
                assert all(c >= 0 and int(c) == c for c in counts)
                n = sum(counts)
                assert n > 0
                centers = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
                mean = sum(c * x for c, x in zip(counts, centers)) / n
                variance = sum(c * (x - mean) ** 2 for c, x in zip(counts, centers)) / n
                rms = math.sqrt(sum(c * x * x for c, x in zip(counts, centers)) / n)
                quantile_intervals = {}
                for q in (0.05, 0.5, 0.95):
                    cumulative = 0
                    for i, count in enumerate(counts):
                        cumulative += count
                        if cumulative >= q * n:
                            quantile_intervals[str(q)] = [edges[i], edges[i + 1]]
                            break
                samples.append({
                    "history_record_clocks": clocks,
                    "bin_edges": edges,
                    "bin_counts": counts,
                    "recorded_count": n,
                    "mean_midpoint_estimate": mean,
                    "std_midpoint_estimate": math.sqrt(variance),
                    "rms_midpoint_estimate": rms,
                    "quantile_bin_intervals": quantile_intervals,
                })
                keys.add(key)
    assert len(keys) == 1 and samples
    assert before == sha256(source), "The source changed while being read."
    result = {
        "schema_version": "0.1",
        "purpose": "Bounded export example, not the production comparison format.",
        "source": rel(source),
        "source_sha256": before,
        "original_parameter_key": next(iter(keys)),
        "canonical_parameter_key": "parameters/" + SUFFIX,
        "semantics": {
            "bin_edges_and_counts": "Original logged histogram values, without rebinning.",
            "moments": "Midpoint estimates, not exact tensor statistics.",
            "quantiles": "Containing-bin intervals for the logged finite-value population.",
            "clocks": "Enclosing W&B history-record coordinates, not exact optimizer-hook timestamps.",
            "amp": "Parameter values are not multiplied by the gradient scaler.",
            "coordinate_identity": "Individual tensor element values are not recoverable.",
        },
        "samples": samples,
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": rel(OUT),
        "parameter": result["original_parameter_key"],
        "samples": len(samples),
        "epochs": [samples[0]["history_record_clocks"].get("epoch"),
                   samples[-1]["history_record_clocks"].get("epoch")],
        "bins_per_sample": sorted({len(s["bin_counts"]) for s in samples}),
        "recorded_count": sorted({s["recorded_count"] for s in samples}),
        "rms_first": samples[0]["rms_midpoint_estimate"],
        "rms_last": samples[-1]["rms_midpoint_estimate"],
        "source_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
