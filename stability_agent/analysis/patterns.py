"""Heuristic pattern labels for low-vs-high stability comparisons."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class PatternEvidence:
    pattern: str
    confidence: float
    rationale: str


def numeric_labels(labels: list[Any]) -> list[float]:
    out: list[float] = []
    for label in labels:
        if label is None:
            continue
        try:
            out.append(float(label))
        except (TypeError, ValueError):
            continue
    return out


def summarize_radius(payload: dict[str, Any]) -> dict[str, Any]:
    labels = numeric_labels(payload.get("labels", []))
    matches = [bool(x) for x in payload.get("matches", [])]
    true_label = payload.get("true_label")
    true_float = _maybe_float(true_label)
    counter = Counter(labels)
    majority_label, majority_share = None, 0.0
    if labels:
        majority_label, count = counter.most_common(1)[0]
        majority_share = count / len(labels)
    disagreement_direction = None
    mean_shift = None
    if labels and true_float is not None:
        mean_label = mean(labels)
        mean_shift = mean_label - true_float
        if abs(mean_shift) >= 0.35:
            disagreement_direction = "higher" if mean_shift > 0 else "lower"
    return {
        "stability_rate": _maybe_float(payload.get("soft_stability_rate")),
        "true_label": true_float,
        "n": len(payload.get("labels", [])),
        "parsed_n": len(labels),
        "null_label_n": len(payload.get("labels", [])) - len(labels),
        "majority_label": majority_label,
        "majority_share": majority_share,
        "mean_label": mean(labels) if labels else None,
        "mean_shift": mean_shift,
        "disagreement_direction": disagreement_direction,
        "match_rate_observed": (sum(matches) / len(matches)) if matches else None,
        "label_counts": {str(k): v for k, v in sorted(counter.items())},
    }


def tag_patterns(summary: dict[str, Any]) -> list[PatternEvidence]:
    """Assign interpretable hypotheses from aggregate stability features."""

    tags: list[PatternEvidence] = []
    avg = summary.get("avg_stability")
    first = summary.get("first_stability")
    last = summary.get("last_stability")
    majority_share = summary.get("overall_majority_share") or 0.0
    direction = summary.get("overall_disagreement_direction")
    null_rate = summary.get("null_label_rate") or 0.0

    if null_rate >= 0.05:
        tags.append(
            PatternEvidence(
                "parse_or_format_instability",
                min(1.0, null_rate * 3),
                f"{null_rate:.1%} of sampled labels could not be parsed.",
            )
        )

    if avg is not None and avg <= 0.30 and majority_share >= 0.65 and direction:
        if direction == "lower":
            pattern = "added_claims_drive_more_negative_verdict"
            rationale = "Perturbations usually move the verdict below the reference label."
        else:
            pattern = "added_claims_drive_more_positive_verdict"
            rationale = "Perturbations usually move the verdict above the reference label."
        tags.append(PatternEvidence(pattern, min(0.95, majority_share), rationale))

    if avg is not None and last is not None and avg <= 0.30 and last <= 0.30:
        tags.append(
            PatternEvidence(
                "radius_insensitive_disagreement",
                0.80,
                "The example stays unstable across the sampled radius range.",
            )
        )

    if first is not None and last is not None and first >= 0.70 and last <= 0.35:
        tags.append(
            PatternEvidence(
                "knife_edge_dependency",
                0.75,
                "Small additions preserve the label, but larger additions break it.",
            )
        )

    if avg is not None and avg >= 0.80 and majority_share <= 0.50:
        tags.append(
            PatternEvidence(
                "stable_without_single_competing_label",
                0.65,
                "Most perturbations match the reference label and errors are diffuse.",
            )
        )

    return tags


def compare_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find examples that are much less stable in one evidence category/source."""

    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(str(row["example_id"]), []).append(row)

    comparisons: list[dict[str, Any]] = []
    for example_id, items in by_example.items():
        if len(items) < 2:
            continue
        values = [
            (item["source_name"], item.get("avg_stability"))
            for item in items
            if item.get("avg_stability") is not None
        ]
        if len(values) < 2:
            continue
        best_source, best = max(values, key=lambda x: x[1])
        worst_source, worst = min(values, key=lambda x: x[1])
        gap = best - worst
        if gap >= 0.30:
            comparisons.append(
                {
                    "example_id": example_id,
                    "most_stable_source": best_source,
                    "least_stable_source": worst_source,
                    "stability_gap": gap,
                    "pattern": f"{worst_source}_sensitive_instability",
                }
            )
    return sorted(comparisons, key=lambda row: row["stability_gap"], reverse=True)


def compare_source_pairs(rows: list[dict[str, Any]], tolerance: float = 0.05) -> list[dict[str, Any]]:
    """Summarize matched-example stability gaps for every pair of sources."""

    by_example: dict[str, dict[str, float]] = {}
    for row in rows:
        value = row.get("avg_stability")
        if value is None:
            continue
        by_example.setdefault(str(row["example_id"]), {})[row["source_name"]] = float(value)

    sources = sorted({row["source_name"] for row in rows})
    summaries: list[dict[str, Any]] = []
    for i, left in enumerate(sources):
        for right in sources[i + 1 :]:
            deltas = [
                values[left] - values[right]
                for values in by_example.values()
                if left in values and right in values
            ]
            if not deltas:
                continue
            summaries.append(
                {
                    "left_source": left,
                    "right_source": right,
                    "n": len(deltas),
                    "mean_delta_left_minus_right": mean(deltas),
                    "median_delta_left_minus_right": median(deltas),
                    "mean_abs_delta": mean(abs(delta) for delta in deltas),
                    "left_more_stable_n": sum(1 for delta in deltas if delta > tolerance),
                    "right_more_stable_n": sum(1 for delta in deltas if delta < -tolerance),
                    "similar_n": sum(1 for delta in deltas if abs(delta) <= tolerance),
                    "large_gap_n": sum(1 for delta in deltas if abs(delta) >= 0.30),
                }
            )
    return sorted(summaries, key=lambda row: row["mean_abs_delta"], reverse=True)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
