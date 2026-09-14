"""Read W&B 0.28 local run files through a strictly read-only descriptor."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

GRAPH_RE = re.compile(r"^(parameters|gradients)/graph_\d+(.*)$")
TRIAL_RE = re.compile(r"trial_(\d+)\.wandb$")
CLOCKS = ("epoch", "trainer/global_step", "_step", "_timestamp", "_runtime")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tensor_key(key: str) -> str:
    match = GRAPH_RE.match(key)
    if match is None:
        return key
    return f"{match.group(1)}/{match.group(2).lstrip('.')}"


def histogram_statistics(edges: list[float], counts: list[int]) -> dict[str, Any]:
    if len(edges) != len(counts) + 1 or not counts:
        raise ValueError("Histogram requires one more edge than counts")
    if any(not math.isfinite(value) for value in edges):
        raise ValueError("Histogram edges must be finite")
    if any(left > right for left, right in zip(edges, edges[1:])):
        raise ValueError("Histogram edges must be ordered")
    if any(count < 0 or int(count) != count for count in counts):
        raise ValueError("Histogram counts must be non-negative integers")
    total = int(sum(counts))
    if total <= 0:
        raise ValueError("Histogram must contain observations")

    centers = [(left + right) / 2 for left, right in zip(edges, edges[1:])]
    mean = sum(count * center for count, center in zip(counts, centers)) / total
    second = sum(count * center * center for count, center in zip(counts, centers)) / total
    variance = max(0.0, second - mean * mean)
    quantiles = {}
    for quantile in (0.05, 0.5, 0.95):
        target = quantile * total
        cumulative = 0
        for index, count in enumerate(counts):
            cumulative += count
            if cumulative >= target:
                quantiles[str(quantile)] = [edges[index], edges[index + 1]]
                break

    epsilon = max(abs(edges[0]), abs(edges[-1]), 1.0) * 1e-3
    certainly_near = sum(
        count for left, right, count in zip(edges, edges[1:], counts)
        if left >= -epsilon and right <= epsilon
    )
    possibly_near = sum(
        count for left, right, count in zip(edges, edges[1:], counts)
        if right >= -epsilon and left <= epsilon
    )
    return {
        "recorded_count": total,
        "range": [edges[0], edges[-1]],
        "mean_midpoint_estimate": mean,
        "std_midpoint_estimate": math.sqrt(variance),
        "rms_midpoint_estimate": math.sqrt(second),
        "l2_midpoint_estimate": math.sqrt(total * second),
        "quantile_bin_intervals": quantiles,
        "near_zero_epsilon": epsilon,
        "near_zero_fraction_bounds": [certainly_near / total, possibly_near / total],
    }


def _cdf_at(value: float, edges: list[float], counts: list[int]) -> float:
    total = sum(counts)
    cumulative = 0.0
    for left, right, count in zip(edges, edges[1:], counts):
        if value >= right:
            cumulative += count
            continue
        if value <= left:
            break
        width = right - left
        cumulative += count if width == 0 else count * (value - left) / width
        break
    return cumulative / total


def normalized_cdf_distance(first: tuple[list[float], list[int]],
                            last: tuple[list[float], list[int]]) -> float:
    points = sorted(set(first[0] + last[0]))
    support = points[-1] - points[0]
    if support == 0:
        return 0.0
    area = 0.0
    previous = points[0]
    previous_delta = abs(_cdf_at(previous, *first) - _cdf_at(previous, *last))
    for point in points[1:]:
        delta = abs(_cdf_at(point, *first) - _cdf_at(point, *last))
        area += (point - previous) * (previous_delta + delta) / 2
        previous, previous_delta = point, delta
    return area / support


def find_sessions(root: Path, experiment: Path) -> dict[int, list[Path]]:
    if not root.is_dir():
        return {}
    slug = "-".join(experiment.parts[-2:])
    result: dict[int, list[Path]] = defaultdict(list)
    for path in root.rglob("run-*.wandb"):
        match = TRIAL_RE.search(path.name)
        if match and (slug in path.name or experiment.name in path.name):
            result[int(match.group(1))].append(path)
    for paths in result.values():
        paths.sort(key=lambda item: (item.stat().st_mtime_ns, str(item)))
    return dict(result)


def read_session(path: Path, include_gradients: bool = False,
                 preserve_raw_histograms: bool = False) -> dict[str, Any]:
    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as error:
        return {"status": "unavailable", "source": str(path), "reason": str(error)}

    before = file_sha256(path)
    datastore = DataStore()
    datastore._fname = str(path)
    datastore._index = 0
    datastore._size_bytes = path.stat().st_size
    datastore._opened_for_scan = True
    datastore._fp = path.open("rb")
    record_counts = Counter()
    versions = set()
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_endpoints: dict[str, list[tuple[list[float], list[int]]]] = defaultdict(list)
    try:
        datastore._read_header()
        while (raw := datastore.scan_data()) is not None:
            record = pb.Record()
            record.ParseFromString(raw)
            kind = record.WhichOneof("record_type")
            record_counts[kind] += 1
            if kind == "telemetry" and record.telemetry.cli_version:
                versions.add(record.telemetry.cli_version)
            if kind != "history":
                continue
            row: dict[str, Any] = {}
            histograms: dict[str, dict[str, Any]] = defaultdict(dict)
            for item in record.history.item:
                parts = list(item.nested_key) or [item.key]
                value = json.loads(item.value_json)
                if len(parts) == 1:
                    row[parts[0]] = value
                    if isinstance(value, dict) and value.get("_type") == "histogram":
                        histograms[parts[0]] = value
                elif len(parts) == 2 and parts[0].startswith(("parameters/", "gradients/")):
                    histograms[parts[0]][parts[1]] = value
            clocks = {key: row.get(key) for key in CLOCKS}
            for original_key, histogram in histograms.items():
                if histogram.get("_type") != "histogram":
                    continue
                if original_key.startswith("gradients/") and not include_gradients:
                    continue
                edges = histogram.get("bins", [])
                counts = histogram.get("values", [])
                try:
                    statistics = histogram_statistics(edges, counts)
                except ValueError:
                    continue
                key = canonical_tensor_key(original_key)
                sample = {
                    "history_record_clocks": clocks,
                    **statistics,
                }
                if preserve_raw_histograms:
                    sample["bin_edges"] = edges
                    sample["bin_counts"] = counts
                series[key].append(sample)
                if len(raw_endpoints[key]) == 0:
                    raw_endpoints[key].append((edges, counts))
                elif len(raw_endpoints[key]) == 1:
                    raw_endpoints[key].append((edges, counts))
                else:
                    raw_endpoints[key][-1] = (edges, counts)
    finally:
        datastore._fp.close()

    summaries = {}
    for key, samples in series.items():
        first, last = samples[0], samples[-1]
        first_rms, last_rms = first["rms_midpoint_estimate"], last["rms_midpoint_estimate"]
        summaries[key] = {
            "samples": len(samples),
            "first": first,
            "last": last,
            "rms_relative_change": (last_rms - first_rms) / first_rms if first_rms else None,
            "mean_change": last["mean_midpoint_estimate"] - first["mean_midpoint_estimate"],
            "normalized_cdf_distance_estimate": normalized_cdf_distance(
                raw_endpoints[key][0], raw_endpoints[key][-1]
            ),
            "trajectory": samples,
        }
    after = file_sha256(path)
    if before != after:
        raise RuntimeError(f"W&B source changed while being read: {path}")
    return {
        "status": "available",
        "source": str(path),
        "source_sha256": before,
        "source_unchanged": True,
        "bytes": path.stat().st_size,
        "record_counts": dict(record_counts),
        "producer_wandb_versions": sorted(versions),
        "series": summaries,
        "semantics": {
            "moments": "Midpoint estimates from logged finite-value histogram bins.",
            "quantiles": "Containing-bin intervals, not exact quantiles.",
            "cdf_distance": "Normalized area between uniform-within-bin CDF estimates.",
            "parameter_amp": "Parameter values are not multiplied by the AMP gradient scaler.",
            "gradient_limit": "Gradient histograms may be scaled, accumulated, pre-clipping microbatch gradients.",
            "coordinate_identity": "Individual tensor coordinates and exact updates are unavailable.",
        },
    }


def reconcile_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    seen: dict[tuple[str, tuple[Any, ...]], list[tuple[int, float, str]]] = defaultdict(list)
    duplicates, conflict_count, conflicts = 0, 0, []
    for session_index, session in enumerate(sessions):
        if session.get("status") != "available":
            continue
        for key, summary in session.get("series", {}).items():
            for sample in summary.get("trajectory", []):
                clocks = sample["history_record_clocks"]
                coordinate = tuple(clocks.get(name) for name in CLOCKS[:3])
                identity = (key, coordinate)
                value = sample["rms_midpoint_estimate"]
                for previous_index, previous_value, previous_source in seen[identity]:
                    if previous_index == session_index:
                        continue
                    if math.isclose(value, previous_value, rel_tol=1e-9, abs_tol=1e-12):
                        duplicates += 1
                    else:
                        conflict_count += 1
                    if not math.isclose(value, previous_value, rel_tol=1e-9, abs_tol=1e-12) and len(conflicts) < 20:
                        conflicts.append({
                            "tensor": key,
                            "history_record_clocks": clocks,
                            "rms_midpoint_estimates": [previous_value, value],
                            "sources": [previous_source, session["source"]],
                        })
                seen[identity].append((session_index, value, session["source"]))
    return {
        "policy": "Sessions remain separate; no conflicting observations are averaged.",
        "cross_session_duplicate_observations": duplicates,
        "cross_session_conflicts": conflict_count,
        "conflict_examples": conflicts,
    }


def read_trials(root: Path | None, experiment: Path, trial_numbers: Iterable[int],
                include_gradients: bool = False,
                preserve_raw_histograms: bool = False) -> dict[str, Any]:
    if root is None:
        return {"status": "unavailable", "reason": "W&B root not provided", "trials": {}}
    requested_numbers = list(trial_numbers)
    sessions = find_sessions(root, experiment)
    result = {}
    for number in requested_numbers:
        paths = sessions.get(number, [])
        decoded_sessions = []
        for path in paths:
            try:
                decoded_sessions.append(
                    read_session(path, include_gradients, preserve_raw_histograms)
                )
            except Exception as error:
                decoded_sessions.append({
                    "status": "unavailable",
                    "source": str(path),
                    "reason": f"{type(error).__name__}: {error}",
                })
        result[str(number)] = {
            "sessions": decoded_sessions,
            "session_count": len(paths),
            "reconciliation": reconcile_sessions(decoded_sessions),
        }
    return {
        "status": "available",
        "root": str(root),
        "trials": result,
        "coverage": {
            "requested_trials": len(requested_numbers),
            "trials_with_sessions": sum(bool(value["session_count"]) for value in result.values()),
            "sessions": sum(value["session_count"] for value in result.values()),
        },
    }
