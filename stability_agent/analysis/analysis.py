"""Core analysis for soft-stability result directories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ..io import StabilityRun, iter_stability_runs
from .patterns import compare_source_pairs, compare_sources, numeric_labels, summarize_radius, tag_patterns
from .redundancy import infer_mask_redundancy, redundancy_summary
from .semantic_redundancy import analyze_support_texts, semantic_summary


@dataclass(frozen=True)
class AnalysisConfig:
    low_threshold: float = 0.30
    high_threshold: float = 0.80
    top_k: int = 12


class StabilityAnalyzer:
    """Analyze low/high stability examples and candidate reasoning patterns."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()

    def analyze_paths(
        self,
        paths: list[Path],
        support_texts_by_index: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        runs = iter_stability_runs(paths)
        rows = [self.summarize_run(run, support_texts_by_index) for run in runs]
        low = [row for row in rows if _lte(row.get("avg_stability"), self.config.low_threshold)]
        high = [row for row in rows if _gte(row.get("avg_stability"), self.config.high_threshold)]
        comparisons = compare_sources(rows)
        return {
            "config": {
                "low_threshold": self.config.low_threshold,
                "high_threshold": self.config.high_threshold,
                "top_k": self.config.top_k,
            },
            "n_runs": len(rows),
            "n_low": len(low),
            "n_high": len(high),
            "source_summary": self.source_summary(rows),
            "source_pair_summary": compare_source_pairs(rows),
            "radius_summary": self.radius_summary(rows),
            "pattern_summary": self.pattern_summary(rows),
            "redundancy_summary": redundancy_summary(rows),
            "semantic_redundancy_summary": semantic_summary(rows),
            "low_examples": sorted(low, key=lambda row: row.get("avg_stability") or 0.0)[
                : self.config.top_k
            ],
            "high_examples": sorted(
                high, key=lambda row: row.get("avg_stability") or 0.0, reverse=True
            )[: self.config.top_k],
            "source_sensitivity": comparisons[: self.config.top_k],
            "rows": rows,
        }

    def summarize_run(
        self,
        run: StabilityRun,
        support_texts_by_index: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        radius_summaries = {
            radius: summarize_radius(payload)
            for radius, payload in sorted(run.radii.items())
            if isinstance(payload, dict)
        }
        rates = [
            payload["stability_rate"]
            for payload in radius_summaries.values()
            if payload.get("stability_rate") is not None
        ]
        first_radius = min(radius_summaries) if radius_summaries else None
        last_radius = max(radius_summaries) if radius_summaries else None
        all_labels: list[Any] = []
        null_labels = 0
        total_labels = 0
        true_label = None
        for payload in run.radii.values():
            if not isinstance(payload, dict):
                continue
            labels = payload.get("labels", [])
            total_labels += len(labels)
            parsed = numeric_labels(labels)
            null_labels += len(labels) - len(parsed)
            all_labels.extend(parsed)
            if true_label is None:
                true_label = payload.get("true_label")

        label_counts = Counter(all_labels)
        majority_label, majority_share = None, 0.0
        if all_labels:
            majority_label, count = label_counts.most_common(1)[0]
            majority_share = count / len(all_labels)
        direction = _direction(all_labels, true_label)
        row = {
            "example_id": run.example_id,
            "source_name": run.source_name,
            "path": str(run.path),
            **infer_mask_redundancy(run.radii),
            "radius_count": len(radius_summaries),
            "avg_stability": mean(rates) if rates else None,
            "min_stability": min(rates) if rates else None,
            "max_stability": max(rates) if rates else None,
            "first_radius": first_radius,
            "first_stability": radius_summaries.get(first_radius, {}).get("stability_rate")
            if first_radius is not None
            else None,
            "last_radius": last_radius,
            "last_stability": radius_summaries.get(last_radius, {}).get("stability_rate")
            if last_radius is not None
            else None,
            "true_label": _maybe_float(true_label),
            "overall_majority_label": majority_label,
            "overall_majority_share": majority_share,
            "overall_disagreement_direction": direction,
            "null_label_rate": (null_labels / total_labels) if total_labels else 0.0,
            "label_counts": {str(k): v for k, v in sorted(label_counts.items())},
            "radii": radius_summaries,
        }
        if support_texts_by_index and run.example_id in support_texts_by_index:
            support_info = support_texts_by_index[run.example_id]
            row.update(analyze_support_texts(support_info.get("support_texts", [])))
            row["semantic_problem_id"] = support_info.get("problem_id")
            row["semantic_source_path"] = support_info.get("path")
        row["patterns"] = [tag.__dict__ for tag in tag_patterns(row)]
        return row

    def source_summary(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_source.setdefault(row["source_name"], []).append(row)
        summary = []
        for source, items in sorted(by_source.items()):
            rates = [x["avg_stability"] for x in items if x.get("avg_stability") is not None]
            summary.append(
                {
                    "source_name": source,
                    "n": len(items),
                    "avg_stability": mean(rates) if rates else None,
                    "low_n": sum(
                        1
                        for x in items
                        if _lte(x.get("avg_stability"), self.config.low_threshold)
                    ),
                    "high_n": sum(
                        1
                        for x in items
                        if _gte(x.get("avg_stability"), self.config.high_threshold)
                    ),
                }
            )
        return summary

    def radius_summary(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_radius: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            for radius, payload in row.get("radii", {}).items():
                if payload.get("stability_rate") is None:
                    continue
                by_radius.setdefault(int(radius), []).append(
                    {
                        "stability_rate": payload["stability_rate"],
                        "source_name": row["source_name"],
                        "example_id": row["example_id"],
                        "direction": payload.get("disagreement_direction"),
                        "majority_share": payload.get("majority_share"),
                    }
                )

        summary: list[dict[str, Any]] = []
        for radius, items in sorted(by_radius.items()):
            rates = [item["stability_rate"] for item in items]
            low = [
                item
                for item in items
                if _lte(item.get("stability_rate"), self.config.low_threshold)
            ]
            high = [
                item
                for item in items
                if _gte(item.get("stability_rate"), self.config.high_threshold)
            ]
            lower = [item for item in low if item.get("direction") == "lower"]
            higher = [item for item in low if item.get("direction") == "higher"]
            summary.append(
                {
                    "radius": radius,
                    "n": len(items),
                    "avg_stability": mean(rates) if rates else None,
                    "low_n": len(low),
                    "low_fraction": len(low) / len(items) if items else None,
                    "high_n": len(high),
                    "high_fraction": len(high) / len(items) if items else None,
                    "low_direction_lower_n": len(lower),
                    "low_direction_higher_n": len(higher),
                    "low_direction_none_n": len(low) - len(lower) - len(higher),
                    "low_majority_share": mean(
                        [
                            item["majority_share"]
                            for item in low
                            if item.get("majority_share") is not None
                        ]
                    )
                    if any(item.get("majority_share") is not None for item in low)
                    else None,
                }
            )
        return summary

    def pattern_summary(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counter: Counter[str] = Counter()
        low_counter: Counter[str] = Counter()
        high_counter: Counter[str] = Counter()
        for row in rows:
            patterns = [p["pattern"] for p in row.get("patterns", [])]
            counter.update(patterns)
            if _lte(row.get("avg_stability"), self.config.low_threshold):
                low_counter.update(patterns)
            if _gte(row.get("avg_stability"), self.config.high_threshold):
                high_counter.update(patterns)
        names = sorted(counter)
        return [
            {
                "pattern": name,
                "n": counter[name],
                "low_n": low_counter[name],
                "high_n": high_counter[name],
            }
            for name in names
        ]


def _direction(labels: list[float], true_label: Any) -> str | None:
    true_float = _maybe_float(true_label)
    if not labels or true_float is None:
        return None
    shift = mean(labels) - true_float
    if abs(shift) < 0.35:
        return None
    return "higher" if shift > 0 else "lower"


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
