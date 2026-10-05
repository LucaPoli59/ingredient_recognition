"""Statistic-consistent inclusion evidence, independent of the frozen D4 rule.

Scores are used as supplied. Exact-image cluster IDs must be recovered and
aligned by the caller; clusters are ordered by first occurrence. A bootstrap
draw samples G clusters G times, carrying every record of each sampled cluster
with its multiplicity. Every checkpoint and prevalence use that same draw.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


def _prepare_inputs(targets, scores, cluster_ids):
    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=np.float64)
    clusters = list(cluster_ids)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets must be a nonempty one-dimensional vector")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("targets must contain only binary values")
    if scores.shape != (5, targets.size) or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite [5, records] array")
    if len(clusters) != targets.size:
        raise ValueError("cluster IDs must align one-to-one with records")
    lookup = {}
    indices = []
    for value in clusters:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            raise ValueError("cluster IDs cannot be missing or non-finite")
        try:
            if value not in lookup:
                lookup[value] = len(lookup)
            indices.append(lookup[value])
        except TypeError as error:
            raise ValueError("cluster IDs must be hashable scalars") from error
    return targets.astype(np.float64), scores, np.asarray(indices, dtype=np.intp), len(lookup)


def _prepare_rankings(targets, scores):
    prepared = []
    for checkpoint_scores in scores:
        order = np.argsort(-checkpoint_scores, kind="stable")
        ranked_scores = checkpoint_scores[order]
        ends = np.r_[np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]), len(order) - 1]
        prepared.append((order, ends, targets[order]))
    return prepared


def _weighted_aps(prepared, weights):
    """Weighted non-interpolated AP, summing precision at each tied-score end."""
    estimates = np.empty((len(weights), len(prepared)), dtype=np.float64)
    for index, (order, ends, ranked_targets) in enumerate(prepared):
        ranked_weights = weights[:, order]
        positive = np.cumsum(ranked_weights * ranked_targets, axis=1)[:, ends]
        total = np.cumsum(ranked_weights, axis=1)[:, ends]
        precision = np.divide(positive, total, out=np.zeros_like(positive, dtype=np.float64), where=total != 0)
        if len(ends) == 1:
            estimates[:, index] = precision[:, 0]
            continue
        increments = np.diff(positive, axis=1, prepend=0.0)
        estimates[:, index] = np.sum(precision * increments, axis=1) / positive[:, -1]
    return estimates


def paired_cluster_bootstrap(
        targets: Sequence[int],
        scores,
        cluster_ids: Sequence[Any],
        *,
        seed: int,
        samples: int = 1000,
        max_draws: int = 10_000,
        batch_size: int = 32,
) -> dict[str, Any]:
    """Estimate median-of-five AP and paired AP-minus-prevalence intervals.

    Percentiles are nominal 2.5% and 97.5% endpoints from ``samples`` valid
    draws. Draws lacking either target class are rejected and counted. If the
    budget is exhausted, partial draws do not produce usable intervals.
    Batching bounds memory and does not change the seeded sample sequence or
    reported attempt count. No logits are averaged or transformed here.
    """
    for name, value, minimum in (("seed", seed, 0), ("samples", samples, 1),
                                 ("max_draws", max_draws, 1), ("batch_size", batch_size, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    targets, scores, cluster_indices, cluster_count = _prepare_inputs(targets, scores, cluster_ids)
    record_count = len(targets)
    prevalence = float(targets.mean())
    result = {
        "status": "insufficient_classes", "bootstrap_valid": False,
        "record_count": record_count, "cluster_count": cluster_count,
        "positive_count": int(targets.sum()), "seed": int(seed),
        "requested_valid_draws": int(samples), "max_draws": int(max_draws),
        "valid_draws": 0, "attempted_draws": 0, "invalid_draws": 0,
        "checkpoint_ap": None, "q": None, "iqr": None,
        "prevalence": prevalence, "q_minus_prevalence": None,
        "q_lower": None, "q_upper": None, "difference_lower": None, "difference_upper": None,
    }
    if prevalence == 0 or prevalence == 1:
        return result

    prepared = _prepare_rankings(targets, scores)
    point_ap = _weighted_aps(prepared, np.ones((1, record_count), dtype=np.float64))[0]
    point_q = float(np.median(point_ap))
    result.update(checkpoint_ap=point_ap.tolist(), q=point_q,
                  iqr=float(np.quantile(point_ap, .75) - np.quantile(point_ap, .25)),
                  q_minus_prevalence=point_q - prevalence)

    generator = np.random.default_rng(seed)
    q_draws = []
    difference_draws = []
    attempts = 0
    while len(q_draws) < samples and attempts < max_draws:
        count = min(batch_size, max_draws - attempts)
        draws = generator.integers(0, cluster_count, size=(count, cluster_count))
        offsets = np.arange(count, dtype=np.intp)[:, None] * cluster_count
        multiplicities = np.bincount((draws + offsets).ravel(), minlength=count * cluster_count)
        weights = multiplicities.reshape(count, cluster_count)[:, cluster_indices]
        positives = np.sum(weights * targets, axis=1)
        totals = np.sum(weights, axis=1)
        valid_indices = np.flatnonzero((positives > 0) & (positives < totals))
        needed = samples - len(q_draws)
        if len(valid_indices) >= needed:
            valid_indices = valid_indices[:needed]
            attempts += int(valid_indices[-1]) + 1
        else:
            attempts += count
        if len(valid_indices):
            estimates = _weighted_aps(prepared, weights[valid_indices])
            q = np.median(estimates, axis=1)
            sampled_prevalence = positives[valid_indices] / totals[valid_indices]
            q_draws.extend(q.tolist())
            difference_draws.extend((q - sampled_prevalence).tolist())

    result.update(valid_draws=len(q_draws), attempted_draws=attempts,
                  invalid_draws=attempts - len(q_draws))
    if len(q_draws) != samples:
        result["status"] = "insufficient_valid_draws"
        return result
    q_lower, q_upper = np.quantile(q_draws, [.025, .975])
    difference_lower, difference_upper = np.quantile(difference_draws, [.025, .975])
    result.update(status="valid", bootstrap_valid=True,
                  q_lower=float(q_lower), q_upper=float(q_upper),
                  difference_lower=float(difference_lower), difference_upper=float(difference_upper))
    return result


def classify_inclusion(
        evidence: Mapping[str, Any], *, quality_floor: float = .20, max_iqr: float = .03,
) -> dict[str, Any]:
    """Apply only held-out quality, paired baseline advantage and dispersion gates.

    Independent reasons survive even when another gate fails. An interval with
    upper AP strictly below the floor with passing dispersion supports
    ``below_quality_floor``; overlap, instability or invalid evidence remains
    ``uncertain``. Equality passes quality/IQR but
    a baseline-difference lower bound of zero does not establish improvement.
    """
    if not math.isfinite(quality_floor) or not 0 <= quality_floor <= 1:
        raise ValueError("quality_floor must lie in [0, 1]")
    if not math.isfinite(max_iqr) or not 0 <= max_iqr <= 1:
        raise ValueError("max_iqr must lie in [0, 1]")

    def number(key):
        value = evidence.get(key)
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            return float(value) if math.isfinite(float(value)) else None
        except (TypeError, ValueError):
            return None

    q, iqr = number("q"), number("iqr")
    lower, upper = number("q_lower"), number("q_upper")
    difference_lower, difference_upper = number("difference_lower"), number("difference_upper")
    point_valid = q is not None and 0 <= q <= 1 and iqr is not None and 0 <= iqr <= 1
    interval_valid = (evidence.get("bootstrap_valid") is True and lower is not None and upper is not None
                      and 0 <= lower <= upper <= 1 and difference_lower is not None
                      and difference_upper is not None and -1 <= difference_lower <= difference_upper <= 1)
    reasons = []
    axis_status = {"evidence": "valid" if point_valid and interval_valid else "invalid",
                   "quality": "unavailable", "baseline_advantage": "unavailable",
                   "temporal_dispersion": "unavailable"}
    if not point_valid:
        reasons.append("incomplete_or_invalid_point_evidence")
    if not interval_valid:
        reasons.append("invalid_bootstrap")
    if q is not None and 0 <= q < quality_floor:
        reasons.append("late_validation_ap_below_floor")
    if iqr is not None and 0 <= iqr <= 1:
        axis_status["temporal_dispersion"] = "pass" if iqr <= max_iqr else "uncertain"
        if iqr > max_iqr:
            reasons.append("late_validation_iqr_above_gate")
    if interval_valid:
        if upper < quality_floor:
            axis_status["quality"] = "below_floor"
            reasons.append("quality_interval_below_floor")
        elif lower < quality_floor or q is None or q < quality_floor:
            axis_status["quality"] = "uncertain"
        else:
            axis_status["quality"] = "pass"
        if lower < quality_floor:
            reasons.append("quality_floor_not_cleared_by_lower_bound")
        axis_status["baseline_advantage"] = "pass" if difference_lower > 0 else "uncertain"
        if difference_lower <= 0:
            reasons.append("positive_baseline_advantage_not_cleared")
    included = not reasons
    outcome = "included" if included else "uncertain"
    if point_valid and interval_valid and iqr <= max_iqr and upper < quality_floor:
        outcome = "below_quality_floor"
    return {"included": included, "outcome": outcome, "reasons": reasons, "axis_status": axis_status}
