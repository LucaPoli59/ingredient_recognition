"""Deterministic scientific SVG view of D6 evidence (no decision logic)."""

from __future__ import annotations

import html
import math
from collections import Counter
from typing import Any, Mapping


COLORS = {"included": "#167c69", "uncertain": "#b67a0b", "below_quality_floor": "#667d94"}
LABELS = {"included": "Eligible", "uncertain": "Uncertain", "below_quality_floor": "Below quality floor"}


def render_inclusion_plot(report: Mapping[str, Any]) -> bytes:
    """Render all report rows with Q intervals, prevalence, and supplied decisions.

    This view never classifies a label. Decision colors also depend on paired
    baseline-advantage and dispersion evidence, which are not encoded by the
    two plot coordinates alone. Class-index ordering makes bytes deterministic.
    """
    rows = sorted(report["labels"], key=lambda row: row["class_index"])
    if len(rows) != report["label_count"]:
        raise ValueError("label count does not match report rows")
    if len({row["class_index"] for row in rows}) != len(rows):
        raise ValueError("class indices must be unique")
    counts = Counter(row["decision"]["outcome"] for row in rows)
    if set(counts) - COLORS.keys():
        raise ValueError("unknown inclusion outcome")
    if dict(counts) != report["outcome_counts"]:
        raise ValueError("outcome counts do not match report rows")
    for row in rows:
        stat = row["statistics"]
        for name in ("q", "prevalence"):
            if not isinstance(stat[name], (int, float)) or not math.isfinite(stat[name]) or not 0 <= stat[name] <= 1:
                raise ValueError(f"{name} must be a finite probability")
        lower, upper = stat["q_lower"], stat["q_upper"]
        if (lower is None) != (upper is None):
            raise ValueError("AP interval endpoints must both exist or both be unavailable")
        if lower is not None and not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1):
            raise ValueError("invalid AP interval")

    escape = lambda value: html.escape(str(value), quote=True)
    width, height = 1160, 750
    left, top, plot_width, plot_height = 84, 127, 738, 474
    x_max = max(.2, min(1., math.ceil(max((r["statistics"]["prevalence"] for r in rows), default=0) * 10) / 10))
    x = lambda value: left + plot_width * value / x_max
    y = lambda value: top + plot_height * (1 - value)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">D6 ingredient inclusion: held-out ranking and uncertainty</title>',
        '<desc id="desc">Every label is shown by validation prevalence and median AP across five late checkpoints, with nominal 95% image-cluster bootstrap intervals. Colors reproduce the complete inclusion decision.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#172a3a">',
        '<text x="84" y="40" font-size="24" font-weight="700">Held-out ingredient ranking</text>',
        f'<text x="84" y="67" font-size="14">{len(rows)} labels · EfficientNetV2-S · seed 42 · single run</text>',
        '<text x="84" y="89" font-size="14">Validation AP at epochs 32, 34, 36, 38 and 40; Q is their median.</text>',
    ]
    for tick in range(6):
        value = tick / 5
        py = y(value)
        pieces += [f'<line x1="{left}" y1="{py:.3f}" x2="{left + plot_width}" y2="{py:.3f}" stroke="#e3e8ed"/>',
                   f'<text x="{left - 14}" y="{py + 5:.3f}" text-anchor="end" font-size="12">{value:.1f}</text>']
    for tick in range(int(round(x_max * 10)) + 1):
        value = tick / 10
        px = x(value)
        pieces += [f'<line x1="{px:.3f}" y1="{top}" x2="{px:.3f}" y2="{top + plot_height}" stroke="#edf0f3"/>',
                   f'<text x="{px:.3f}" y="{top + plot_height + 23}" text-anchor="middle" font-size="12">{value:.1f}</text>']
    pieces += [
        f'<line x1="{left}" y1="{y(.2):.3f}" x2="{left + plot_width}" y2="{y(.2):.3f}" stroke="#303e4a" stroke-width="1.5" stroke-dasharray="7 4"/>',
        f'<line x1="{x(0):.3f}" y1="{y(0):.3f}" x2="{x(x_max):.3f}" y2="{y(x_max):.3f}" stroke="#98a3ac" stroke-width="1.3" stroke-dasharray="3 4"/>',
    ]
    for row in rows:
        stat, outcome = row["statistics"], row["decision"]["outcome"]
        px, py, color = x(stat["prevalence"]), y(stat["q"]), COLORS[outcome]
        interval = ("unavailable" if stat["q_lower"] is None else f'{stat["q_lower"]:.4f}–{stat["q_upper"]:.4f}')
        tooltip = escape(f'{row["class_name"]}: Q={stat["q"]:.4f}; 95% interval={interval}; '
                         f'prevalence={stat["prevalence"]:.4f}; decision={outcome}')
        pieces.append(f'<g data-class-index="{row["class_index"]}"><title>{tooltip}</title>')
        if stat["q_lower"] is not None:
            low, high = y(stat["q_lower"]), y(stat["q_upper"])
            pieces += [f'<line x1="{px:.3f}" y1="{low:.3f}" x2="{px:.3f}" y2="{high:.3f}" stroke="{color}" stroke-opacity="0.42" stroke-width="1.2"/>',
                       f'<path d="M {px - 2.5:.3f} {low:.3f} h 5 M {px - 2.5:.3f} {high:.3f} h 5" stroke="{color}" stroke-opacity="0.42" fill="none"/>']
        pieces.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="3.4" fill="{color}" fill-opacity="0.84" stroke="#ffffff" stroke-width="0.55"/></g>')
    pieces += [
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#8c99a4"/>',
        f'<text x="{left + plot_width / 2:.3f}" y="{top + plot_height + 52}" text-anchor="middle" font-size="14">Validation prevalence (constant-score AP)</text>',
        f'<text x="24" y="{top + plot_height / 2:.3f}" transform="rotate(-90 24 {top + plot_height / 2:.3f})" text-anchor="middle" font-size="14">Median checkpoint AP (Q)</text>',
        '<text x="865" y="150" font-size="16" font-weight="700">Complete D6 decision</text>',
    ]
    for index, outcome in enumerate(COLORS):
        py = 179 + 29 * index
        pieces += [f'<circle cx="872" cy="{py - 4}" r="5" fill="{COLORS[outcome]}"/>',
                   f'<text x="887" y="{py}" font-size="13">{LABELS[outcome]}: {counts[outcome]}</text>']
    pieces += [
        '<line x1="866" y1="279" x2="898" y2="279" stroke="#303e4a" stroke-width="1.5" stroke-dasharray="7 4"/>',
        '<text x="907" y="283" font-size="12">Quality floor Q = 0.20</text>',
        '<line x1="866" y1="304" x2="898" y2="304" stroke="#98a3ac" stroke-dasharray="3 4"/>',
        '<text x="907" y="308" font-size="12">Q = prevalence</text>',
    ]
    sensitivity = report.get("sensitivity_counts", {})
    if sensitivity:
        pieces.append('<text x="865" y="365" font-size="16" font-weight="700">Fixed floor sensitivity</text>')
        for index, (floor, values) in enumerate(sorted(sensitivity.items(), key=lambda pair: float(pair[0]))):
            if set(values) - COLORS.keys() or sum(values.values()) != len(rows):
                raise ValueError("sensitivity counts do not match report labels")
            py, cursor = 390 + 56 * index, 866.
            pieces.append(f'<text x="866" y="{py}" font-size="12">Q floor {float(floor):.2f}: {values.get("included", 0)} eligible</text>')
            for outcome in COLORS:
                bar_width = 240 * values.get(outcome, 0) / max(len(rows), 1)
                if bar_width:
                    pieces.append(f'<rect x="{cursor:.3f}" y="{py + 8}" width="{bar_width:.3f}" height="13" fill="{COLORS[outcome]}"/>')
                cursor += bar_width
        pieces.append('<text x="865" y="579" font-size="12">Sensitivity does not choose the floor.</text>')
    pieces += [
        '<text x="84" y="686" font-size="12">Vertical bars: nominal 95% paired image-cluster bootstrap intervals for Q.</text>',
        '<text x="84" y="707" font-size="12">Colors also require paired Q − prevalence and IQR gates; plot coordinates alone do not determine eligibility.</text>',
        '<text x="84" y="728" font-size="12">Exploratory post-P4 amendment · no test results · eligibility evidence, not the final P6 vocabulary.</text>',
        '</g></svg>',
    ]
    return ("\n".join(pieces) + "\n").encode("utf-8")
