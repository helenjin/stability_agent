"""Summaries for PopQA-backed source comparison reports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_popqa_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load PopQA metadata from CSV, JSON, or JSONL.

    The output is keyed by PopQA's `id` field when available, with a fallback
    to row index for locally reindexed experiments.
    """

    rows = _load_rows(path)
    subject_pops = [_maybe_float(row.get("s_pop")) for row in rows]
    object_pops = [_maybe_float(row.get("o_pop")) for row in rows]
    subject_pops = [x for x in subject_pops if x is not None]
    object_pops = [x for x in object_pops if x is not None]
    s_cutoffs = _tertiles(subject_pops)
    o_cutoffs = _tertiles(object_pops)

    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        key = str(row.get("id", idx))
        subject_pop = _maybe_float(row.get("s_pop"))
        object_pop = _maybe_float(row.get("o_pop"))
        item = {
            "id": row.get("id", idx),
            "question": row.get("question"),
            "subject": row.get("subj"),
            "property": row.get("prop"),
            "object": row.get("obj"),
            "possible_answers": _parse_answers(row.get("possible_answers")),
            "subject_popularity": subject_pop,
            "object_popularity": object_pop,
            "subject_pop_bucket": _bucket(subject_pop, s_cutoffs),
            "object_pop_bucket": _bucket(object_pop, o_cutoffs),
        }
        out[key] = item
        out.setdefault(str(idx), item)
    return out


def summarize_popqa_source_effects(
    result: dict[str, Any],
    metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Join PopQA metadata onto a source-comparison result."""

    rows = []
    for row in result.get("rows", []):
        meta = metadata_by_id.get(str(row["example_id"]), {})
        enriched = {
            **row,
            "question": meta.get("question"),
            "subject": meta.get("subject"),
            "property": meta.get("property"),
            "object": meta.get("object"),
            "possible_answers": meta.get("possible_answers"),
            "subject_popularity": meta.get("subject_popularity"),
            "object_popularity": meta.get("object_popularity"),
            "subject_pop_bucket": meta.get("subject_pop_bucket"),
            "object_pop_bucket": meta.get("object_pop_bucket"),
        }
        rows.append(enriched)

    return {
        "source_summary": result.get("source_summary", []),
        "source_pair_summary": result.get("source_pair_summary", []),
        "subject_popularity_summary": _popularity_summary(rows, "subject_pop_bucket"),
        "object_popularity_summary": _popularity_summary(rows, "object_pop_bucket"),
        "source_gap_examples": _source_gap_examples(rows),
    }


def render_popqa_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# PopQA Source Follow-up")
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

    lines.append("## Subject Popularity by Source")
    lines.append("")
    lines.append("| subject popularity | source | n | avg stability | low | high |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in summary["subject_popularity_summary"]:
        lines.append(
            f"| {row['bucket']} | {row['source_name']} | {row['n']} | "
            f"{_fmt(row.get('avg_stability'))} | {row['low_n']} | {row['high_n']} |"
        )
    lines.append("")

    lines.append("## Object Popularity by Source")
    lines.append("")
    lines.append("| object popularity | source | n | avg stability | low | high |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in summary["object_popularity_summary"]:
        lines.append(
            f"| {row['bucket']} | {row['source_name']} | {row['n']} | "
            f"{_fmt(row.get('avg_stability'))} | {row['low_n']} | {row['high_n']} |"
        )
    lines.append("")

    lines.append("## Largest Source Gaps")
    lines.append("")
    lines.append("| example | question | subject pop | most stable | least stable | gap |")
    lines.append("| --- | --- | --- | --- | --- | ---: |")
    for row in summary["source_gap_examples"]:
        lines.append(
            f"| {row['example_id']} | {_truncate(row.get('question'))} | "
            f"{row.get('subject_pop_bucket') or '-'} | {row['most_stable_source']} | "
            f"{row['least_stable_source']} | {_fmt(row.get('stability_gap'))} |"
        )
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append(
        "For PopQA, the most important slice is popularity. If retrieval is helpful mainly on "
        "tail entities, the retrieved source conditions should close the gap in low-popularity "
        "buckets even if parametric memory stays strong on popular entities."
    )
    return "\n".join(lines) + "\n"


def _popularity_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = row.get(field) or "unknown"
        grouped[(bucket, row["source_name"])].append(row)

    out = []
    for (bucket, source_name), items in sorted(grouped.items()):
        stability_vals = [item["avg_stability"] for item in items if item.get("avg_stability") is not None]
        out.append(
            {
                "bucket": bucket,
                "source_name": source_name,
                "n": len(items),
                "avg_stability": mean(stability_vals) if stability_vals else None,
                "low_n": sum(1 for item in items if _lte(item.get("avg_stability"), 0.30)),
                "high_n": sum(1 for item in items if _gte(item.get("avg_stability"), 0.80)),
            }
        )
    return out


def _source_gap_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "question": ref.get("question"),
                "subject_pop_bucket": ref.get("subject_pop_bucket"),
                "most_stable_source": best_source,
                "least_stable_source": worst_source,
                "stability_gap": gap,
            }
        )

    examples.sort(key=lambda row: row["stability_gap"], reverse=True)
    return examples[:12]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            return list(data.values())
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported PopQA metadata format: {path}")


def _parse_answers(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return [str(value)]
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [text]


def _tertiles(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 3:
        return None
    ordered = sorted(values)
    first = ordered[len(ordered) // 3]
    second = ordered[(2 * len(ordered)) // 3]
    return (first, second)


def _bucket(value: float | None, cutoffs: tuple[float, float] | None) -> str:
    if value is None or cutoffs is None:
        return "unknown"
    low, high = cutoffs
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def _truncate(value: Any, max_chars: int = 72) -> str:
    text = str(value or "-")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lte(value: Any, threshold: float) -> bool:
    value = _maybe_float(value)
    return value is not None and value <= threshold


def _gte(value: Any, threshold: float) -> bool:
    value = _maybe_float(value)
    return value is not None and value >= threshold
