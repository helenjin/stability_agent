"""Summaries for ClaimSpy-backed source comparison reports."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

# Loader lives with the adapter now; re-exported here for backward compatibility.
from ..datasets.claimspy import load_claimspy_metadata  # noqa: F401


def summarize_claimspy_source_effects(
    result: dict[str, Any],
    metadata_by_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Join ClaimSpy metadata onto a source-comparison result."""

    rows = []
    for row in result.get("rows", []):
        meta = metadata_by_index.get(str(row["example_id"]), {})
        score = meta.get("continuous_score")
        if score is None:
            score = meta.get("likert_score")
        enriched = {
            **row,
            "problem_id": meta.get("problem_id"),
            "domain": meta.get("domain"),
            "quality_score": score,
            "confidence_score": meta.get("confidence"),
        }
        rows.append(enriched)

    result = dict(result)
    result["rows"] = rows

    return {
        "source_summary": result.get("source_summary", []),
        "source_pair_summary": result.get("source_pair_summary", []),
        "domain_source_summary": _domain_source_summary(rows),
        "quality_buckets": _quality_buckets(rows),
        "source_gap_examples": _source_gap_examples(rows, result.get("source_pair_summary", [])),
    }


def render_claimspy_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ClaimSpy Source Follow-up")
    lines.append("")

    lines.append("## Source Summary")
    lines.append("")
    lines.append("| source | avg stability | low | high |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in summary["source_summary"]:
        lines.append(
            f"| {row['source_name']} | {_fmt(row.get('avg_stability'))} | "
            f"{row['low_n']} | {row['high_n']} |"
        )
    lines.append("")

    if summary.get("source_pair_summary"):
        lines.append("## Source Pair Summary")
        lines.append("")
        lines.append(
            "| left source | right source | mean left-right | mean abs gap | left higher | right higher | similar |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in summary["source_pair_summary"]:
            lines.append(
                f"| {row['left_source']} | {row['right_source']} | "
                f"{_fmt(row.get('mean_delta_left_minus_right'))} | "
                f"{_fmt(row.get('mean_abs_delta'))} | {row['left_more_stable_n']} | "
                f"{row['right_more_stable_n']} | {row['similar_n']} |"
            )
        lines.append("")

    lines.append("## Domain by Source")
    lines.append("")
    lines.append("| domain | source | n | avg stability | avg quality | low | high |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["domain_source_summary"]:
        lines.append(
            f"| {row['domain']} | {row['source_name']} | {row['n']} | "
            f"{_fmt(row.get('avg_stability'))} | {_fmt(row.get('avg_quality_score'))} | "
            f"{row['low_n']} | {row['high_n']} |"
        )
    lines.append("")

    lines.append("## Stability vs Quality")
    lines.append("")
    lines.append("| source | bucket | n | avg stability | avg quality |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in summary["quality_buckets"]:
        lines.append(
            f"| {row['source_name']} | {row['bucket']} | {row['n']} | "
            f"{_fmt(row.get('avg_stability'))} | {_fmt(row.get('avg_quality_score'))} |"
        )
    lines.append("")

    lines.append("## Largest Source Gaps")
    lines.append("")
    lines.append("| example | problem | domain | most stable | least stable | gap | quality |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: |")
    for row in summary["source_gap_examples"]:
        lines.append(
            f"| {row['example_id']} | {row.get('problem_id') or '-'} | {row.get('domain') or '-'} | "
            f"{row['most_stable_source']} | {row['least_stable_source']} | "
            f"{_fmt(row.get('stability_gap'))} | {_fmt(row.get('quality_score'))} |"
        )
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append(
        "These summaries are diagnostic. A higher stability score means the judgment changes less "
        "under evidence perturbation; it does not automatically mean the judgment is more correct."
    )
    return "\n".join(lines) + "\n"


def _domain_source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("domain") or "unknown", row["source_name"])].append(row)

    out = []
    for (domain, source_name), items in sorted(grouped.items()):
        stability_vals = [item["avg_stability"] for item in items if item.get("avg_stability") is not None]
        quality_vals = [item["quality_score"] for item in items if item.get("quality_score") is not None]
        out.append(
            {
                "domain": domain,
                "source_name": source_name,
                "n": len(items),
                "avg_stability": mean(stability_vals) if stability_vals else None,
                "avg_quality_score": mean(quality_vals) if quality_vals else None,
                "low_n": sum(1 for item in items if _lte(item.get("avg_stability"), 0.30)),
                "high_n": sum(1 for item in items if _gte(item.get("avg_stability"), 0.80)),
            }
        )
    return out


def _quality_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = _quality_bucket(row.get("quality_score"))
        grouped[(row["source_name"], bucket)].append(row)

    out = []
    for (source_name, bucket), items in sorted(grouped.items()):
        stability_vals = [item["avg_stability"] for item in items if item.get("avg_stability") is not None]
        quality_vals = [item["quality_score"] for item in items if item.get("quality_score") is not None]
        out.append(
            {
                "source_name": source_name,
                "bucket": bucket,
                "n": len(items),
                "avg_stability": mean(stability_vals) if stability_vals else None,
                "avg_quality_score": mean(quality_vals) if quality_vals else None,
            }
        )
    return out


def _source_gap_examples(
    rows: list[dict[str, Any]],
    source_pair_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_example[str(row["example_id"])].append(row)

    examples = []
    for example_id, items in by_example.items():
        values = [
            (item["source_name"], item.get("avg_stability"))
            for item in items
            if item.get("avg_stability") is not None
        ]
        if len(values) < 2:
            continue
        best_source, best = max(values, key=lambda pair: pair[1])
        worst_source, worst = min(values, key=lambda pair: pair[1])
        gap = best - worst
        if gap < 0.30:
            continue
        ref = items[0]
        examples.append(
            {
                "example_id": example_id,
                "problem_id": ref.get("problem_id"),
                "domain": ref.get("domain"),
                "most_stable_source": best_source,
                "least_stable_source": worst_source,
                "stability_gap": gap,
                "quality_score": ref.get("quality_score"),
            }
        )

    examples.sort(key=lambda row: row["stability_gap"], reverse=True)
    return examples[:12]


def _quality_bucket(score: Any) -> str:
    score = _maybe_float(score)
    if score is None:
        return "missing"
    if score >= 0.67:
        return "high_quality"
    if score <= 0.33:
        return "low_quality"
    return "mid_quality"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _lte(value: Any, threshold: float) -> bool:
    value = _maybe_float(value)
    return value is not None and value <= threshold


def _gte(value: Any, threshold: float) -> bool:
    value = _maybe_float(value)
    return value is not None and value >= threshold
