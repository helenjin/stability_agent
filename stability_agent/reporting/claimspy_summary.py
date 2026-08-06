"""Summaries for ClaimSpy-backed source comparison reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

# Loader lives with the adapter now; re-exported here for backward compatibility.
from ..datasets.claimspy import load_claimspy_metadata  # noqa: F401


def _enrich_rows(
    rows: list[dict[str, Any]],
    metadata_by_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join per-source stability rows onto ClaimSpy-shaped metadata, keyed by
    ``str(example_id)`` (the stability run's positional index, ``di``)."""

    enriched = []
    for row in rows:
        meta = metadata_by_index.get(str(row["example_id"]), {})
        score = meta.get("continuous_score")
        if score is None:
            score = meta.get("likert_score")
        enriched.append({
            **row,
            "problem_id": meta.get("problem_id"),
            "domain": meta.get("domain"),
            "quality_score": score,
            "confidence_score": meta.get("confidence"),
        })
    return enriched


def summarize_claimspy_source_effects(
    result: dict[str, Any],
    metadata_by_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Join ClaimSpy metadata onto a source-comparison result."""

    rows = _enrich_rows(result.get("rows", []), metadata_by_index)

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


def build_gap_audit(
    result: dict[str, Any],
    metadata_by_index: dict[str, dict[str, Any]],
    gap_threshold: float = 0.30,
    stable_threshold: float = 0.80,
    collapse_threshold: float = 0.10,
    quality_correct: float = 0.67,
    quality_wrong: float = 0.33,
    parametric_source: str = "ss_rate_dicts_all_parametric_v0",
    top_k: int = 12,
) -> dict[str, Any]:
    """Audit every example with a large cross-source stability gap.

    Extends ``_source_gap_examples`` from a top-``top_k`` list to the full
    population of gap cases, with domain and signature breakdowns. Signatures
    are patterns in the serialized outputs, not verified causal mechanisms.
    """

    rows = _enrich_rows(result.get("rows", []), metadata_by_index)
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_example[str(row["example_id"])].append(row)

    cases: list[dict[str, Any]] = []
    for example_id, items in by_example.items():
        stabs = {
            item["source_name"]: item["avg_stability"]
            for item in items
            if item.get("avg_stability") is not None
        }
        if len(stabs) < 2:
            continue
        best_source, best = max(stabs.items(), key=lambda pair: pair[1])
        worst_source, worst = min(stabs.items(), key=lambda pair: pair[1])
        gap = best - worst
        if gap < gap_threshold:
            continue

        by_source_row = {item["source_name"]: item for item in items}
        ref = items[0]
        quality = _maybe_float(ref.get("quality_score"))
        parametric_stab = stabs.get(parametric_source)

        is_parametric_best = best_source == parametric_source
        if is_parametric_best:
            retrieved = {s: v for s, v in stabs.items() if s != parametric_source}
            collapsed_source, collapsed_val = min(retrieved.items(), key=lambda pair: pair[1])
            collapsed_patterns = {
                p["pattern"] for p in by_source_row[collapsed_source].get("patterns", [])
            }
            if collapsed_val <= collapse_threshold and "radius_insensitive_disagreement" in collapsed_patterns:
                signature = "retrieval_oversensitivity"
            else:
                signature = "parametric_more_stable"
        else:
            signature = "retrieved_more_stable"

        cell = None
        if parametric_stab is not None and parametric_stab >= stable_threshold:
            if quality is None:
                cell = "stable-mid"
            elif quality >= quality_correct:
                cell = "stable-correct"
            elif quality <= quality_wrong:
                cell = "stable-wrong"
            else:
                cell = "stable-mid"

        cases.append({
            "example_id": example_id,
            "problem_id": ref.get("problem_id"),
            "domain": ref.get("domain"),
            "quality_score": quality,
            "stability_by_source": stabs,
            "most_stable_source": best_source,
            "least_stable_source": worst_source,
            "stability_gap": gap,
            "disagreement_direction": by_source_row[worst_source].get("overall_disagreement_direction"),
            "signature": signature,
            "cell": cell,
        })

    cases.sort(key=lambda row: row["stability_gap"], reverse=True)

    param_stabs_in_gaps = [
        c["stability_by_source"].get(parametric_source)
        for c in cases
        if c["stability_by_source"].get(parametric_source) is not None
    ]
    signature_counts = Counter(c["signature"] for c in cases)
    domain_totals: dict[str, list[float]] = defaultdict(list)
    for c in cases:
        domain_totals[c.get("domain") or "unknown"].append(c["stability_gap"])
    domain_summary = [
        {"domain": domain, "n": len(gaps), "mean_gap": mean(gaps)}
        for domain, gaps in domain_totals.items()
    ]
    domain_summary.sort(key=lambda row: row["n"], reverse=True)

    return {
        "config": {
            "gap_threshold": gap_threshold,
            "stable_threshold": stable_threshold,
            "collapse_threshold": collapse_threshold,
            "quality_correct": quality_correct,
            "quality_wrong": quality_wrong,
            "parametric_source": parametric_source,
        },
        "n_total": len(by_example),
        "n_gap_cases": len(cases),
        "most_stable_source_counts": dict(Counter(c["most_stable_source"] for c in cases)),
        "parametric_stability_in_gap_cases": {
            "mean": mean(param_stabs_in_gaps) if param_stabs_in_gaps else None,
            "n_high_ge_0.80": sum(1 for v in param_stabs_in_gaps if v >= 0.80),
            "n_mid_0.50_0.80": sum(1 for v in param_stabs_in_gaps if 0.50 <= v < 0.80),
            "n_low_lt_0.50": sum(1 for v in param_stabs_in_gaps if v < 0.50),
        },
        "disagreement_direction_counts": dict(
            Counter(c["disagreement_direction"] for c in cases if c["disagreement_direction"])
        ),
        "signature_counts": dict(signature_counts),
        "domain_summary": domain_summary,
        "top_gaps": cases[:top_k],
        "stable_wrong_cases": [c for c in cases if c["cell"] == "stable-wrong"],
        "counter_examples_retrieved_more_stable": [
            c for c in cases if c["signature"] == "retrieved_more_stable"
        ],
    }


def render_gap_audit_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    cfg = audit["config"]
    param = cfg["parametric_source"]
    sources = sorted({
        s
        for c in audit["top_gaps"] + audit["counter_examples_retrieved_more_stable"]
        for s in c["stability_by_source"]
    })
    lines.append("# ClaimSpy Large-Gap Audit (Systematic)")
    lines.append("")
    lines.append(
        f"Of {audit['n_total']} matched examples, **{audit['n_gap_cases']} have a cross-source "
        f"stability gap >= {cfg['gap_threshold']:.2f}** (gap = max - min avg_stability across sources)."
    )
    lines.append("")

    pstab = audit["parametric_stability_in_gap_cases"]
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- `{param}` is the most stable source in "
        f"{audit['most_stable_source_counts'].get(param, 0)} of {audit['n_gap_cases']} gap cases "
        f"(mean stability {_fmt(pstab['mean'])}; >= 0.80 in {pstab['n_high_ge_0.80']}, "
        f"0.50-0.80 in {pstab['n_mid_0.50_0.80']}, < 0.50 in {pstab['n_low_lt_0.50']})."
    )
    for source, n in audit["most_stable_source_counts"].items():
        if source != param:
            lines.append(f"- `{source}` is the most stable source in {n} gap cases.")
    lines.append(
        "- Disagreement direction on the least-stable side: "
        + ", ".join(f"`{d}` in {n}" for d, n in audit["disagreement_direction_counts"].items())
        + "."
    )
    lines.append("")

    lines.append("## Signatures (data patterns, not confirmed causes)")
    lines.append("")
    lines.append("| signature | n |")
    lines.append("| --- | ---: |")
    for sig, n in sorted(audit["signature_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {sig} | {n} |")
    lines.append("")
    lines.append(
        "> Caveat: these are signatures in the serialized outputs, not verified mechanisms."
    )
    lines.append("")

    lines.append("## By Domain")
    lines.append("")
    lines.append("| domain | gap cases | mean gap |")
    lines.append("| --- | ---: | ---: |")
    for row in audit["domain_summary"]:
        lines.append(f"| {row['domain']} | {row['n']} | {_fmt(row['mean_gap'])} |")
    lines.append("")

    lines.append(f"## Top {len(audit['top_gaps'])} Largest Gaps")
    lines.append("")
    lines.append("| ex | problem | domain | q | " + " | ".join(sources) + " | gap | dir | cell |")
    lines.append("| ---: | --- | --- | ---: | " + " | ".join(["---:"] * len(sources)) + " | ---: | --- | --- |")
    for c in audit["top_gaps"]:
        stab_cells = " | ".join(_fmt(c["stability_by_source"].get(s)) for s in sources)
        lines.append(
            f"| {c['example_id']} | {c.get('problem_id') or '-'} | {c.get('domain') or '-'} | "
            f"{_fmt(c['quality_score'])} | {stab_cells} | {_fmt(c['stability_gap'])} | "
            f"{c.get('disagreement_direction') or '-'} | {c.get('cell') or '-'} |"
        )
    lines.append("")

    lines.append("## Stable-Wrong Cases (parametric stable, quality low)")
    lines.append("")
    lines.append(
        "Parametric reasoning that is perturbation-robust **and** low-quality -- rigidity, not "
        "reliability."
    )
    lines.append("")
    lines.append("| ex | problem | domain | param | quality | dir |")
    lines.append("| ---: | --- | --- | ---: | ---: | --- |")
    for c in audit["stable_wrong_cases"]:
        lines.append(
            f"| {c['example_id']} | {c.get('problem_id') or '-'} | {c.get('domain') or '-'} | "
            f"{_fmt(c['stability_by_source'].get(param))} | {_fmt(c['quality_score'])} | "
            f"{c.get('disagreement_direction') or '-'} |"
        )
    lines.append("")

    counter = audit["counter_examples_retrieved_more_stable"]
    lines.append(f"## Counter-Examples: Retrieval More Stable ({len(counter)})")
    lines.append("")
    lines.append("| ex | problem | domain | q | " + " | ".join(sources) + " | best |")
    lines.append("| ---: | --- | --- | ---: | " + " | ".join(["---:"] * len(sources)) + " | --- |")
    for c in counter:
        stab_cells = " | ".join(_fmt(c["stability_by_source"].get(s)) for s in sources)
        lines.append(
            f"| {c['example_id']} | {c.get('problem_id') or '-'} | {c.get('domain') or '-'} | "
            f"{_fmt(c['quality_score'])} | {stab_cells} | {c['most_stable_source']} |"
        )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Quality label is a proxy. ClaimSpy `continuous_score` is treated as ground-truth "
        "correctness; it is itself a model-assisted judgment."
    )
    lines.append(
        "- Signatures are not causes. The error-mode taxonomy (stale memory, conflicting/noisy "
        "retrieval, sparse support, multi-hop brittleness) cannot be confirmed from serialized "
        "labels alone."
    )
    return "\n".join(lines) + "\n"


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
