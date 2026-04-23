"""Redundancy-style proxy metrics from perturbation masks."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any


REDUNDANCY_METRICS = [
    "total_claims",
    "base_claims",
    "addable_claims",
    "base_fraction",
    "avg_selected",
    "avg_selected_fraction",
]


def infer_mask_redundancy(radii: dict[int, dict[str, Any]]) -> dict[str, float | int | None]:
    """Infer simple redundancy proxies from sampled alpha perturbations.

    These are structural proxies, not semantic redundancy judgments. The
    always-present claims approximate the original mask because perturbations
    only flip zeros to ones.
    """

    if not radii:
        return _empty()
    first_radius = min(radii)
    rows = radii.get(first_radius, {}).get("alpha_pertbs", [])
    if not rows:
        return _empty()

    total_claims = len(rows[0])
    if total_claims == 0:
        return _empty()
    base_claims = sum(1 for col in range(total_claims) if all(row[col] == 1 for row in rows))
    avg_selected = mean(sum(row) for row in rows)
    return {
        "total_claims": total_claims,
        "base_claims": base_claims,
        "addable_claims": total_claims - base_claims,
        "base_fraction": base_claims / total_claims,
        "avg_selected": avg_selected,
        "avg_selected_fraction": avg_selected / total_claims,
    }


def redundancy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "correlation_with_avg_stability": {
            metric: pearson(
                [row.get(metric) for row in rows],
                [row.get("avg_stability") for row in rows],
            )
            for metric in REDUNDANCY_METRICS
        },
        "correlation_with_radius_insensitive_disagreement": {
            metric: pearson(
                [row.get(metric) for row in rows],
                [_has_pattern(row, "radius_insensitive_disagreement") for row in rows],
            )
            for metric in REDUNDANCY_METRICS
        },
        "group_means": {
            "radius_insensitive_disagreement": _means(
                [row for row in rows if _has_pattern(row, "radius_insensitive_disagreement")]
            ),
            "non_radius_insensitive_disagreement": _means(
                [row for row in rows if not _has_pattern(row, "radius_insensitive_disagreement")]
            ),
            "low_stability": _means(
                [row for row in rows if _lte(row.get("avg_stability"), 0.30)]
            ),
            "high_stability": _means(
                [row for row in rows if _gte(row.get("avg_stability"), 0.80)]
            ),
        },
    }


def pearson(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None and not _is_nan(x) and not _is_nan(y)
    ]
    if len(pairs) < 3:
        return None
    x_vals, y_vals = zip(*pairs)
    x_mean = mean(x_vals)
    y_mean = mean(y_vals)
    x_var = sum((x - x_mean) ** 2 for x in x_vals)
    y_var = sum((y - y_mean) ** 2 for y in y_vals)
    if x_var == 0 or y_var == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / math.sqrt(x_var * y_var)


def _means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for metric in REDUNDANCY_METRICS:
        vals = [row.get(metric) for row in rows if row.get(metric) is not None]
        out[metric] = mean(vals) if vals else None
    return out


def _has_pattern(row: dict[str, Any], pattern: str) -> int:
    return int(any(item.get("pattern") == pattern for item in row.get("patterns", [])))


def _empty() -> dict[str, None]:
    return {metric: None for metric in REDUNDANCY_METRICS}


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _lte(value: Any, threshold: float) -> bool:
    return value is not None and float(value) <= threshold


def _gte(value: Any, threshold: float) -> bool:
    return value is not None and float(value) >= threshold
