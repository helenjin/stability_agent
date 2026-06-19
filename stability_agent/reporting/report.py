"""Markdown report rendering."""

from __future__ import annotations

from typing import Any


def render_markdown_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    cfg = result["config"]
    lines.append("# Stability Agent Report")
    lines.append("")
    lines.append(
        f"Analyzed {result['n_runs']} runs with low <= {cfg['low_threshold']} "
        f"and high >= {cfg['high_threshold']}."
    )
    lines.append("")
    lines.append(f"- Low-stability runs: {result['n_low']}")
    lines.append(f"- High-stability runs: {result['n_high']}")
    lines.append("")

    lines.append("## Source Summary")
    lines.append("")
    lines.append("| source | n | avg stability | low | high |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in result["source_summary"]:
        lines.append(
            f"| {row['source_name']} | {row['n']} | {_fmt(row['avg_stability'])} | "
            f"{row['low_n']} | {row['high_n']} |"
        )
    lines.append("")

    if result.get("source_pair_summary"):
        lines.append("## Source Pair Summary")
        lines.append("")
        lines.append(
            "| left source | right source | n | mean left-right | median left-right | "
            "mean abs gap | left higher | right higher | similar | large gap |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in result["source_pair_summary"]:
            lines.append(
                f"| {row['left_source']} | {row['right_source']} | {row['n']} | "
                f"{_fmt(row.get('mean_delta_left_minus_right'))} | "
                f"{_fmt(row.get('median_delta_left_minus_right'))} | "
                f"{_fmt(row.get('mean_abs_delta'))} | "
                f"{row['left_more_stable_n']} | {row['right_more_stable_n']} | "
                f"{row['similar_n']} | {row['large_gap_n']} |"
            )
        lines.append("")

    lines.append("## Radius Summary")
    lines.append("")
    if result.get("radius_summary"):
        lines.append(
            "| radius | n | avg stability | low | low frac | high | high frac | "
            "low lower | low higher | low no dir | low majority share |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in result["radius_summary"]:
            lines.append(
                f"| {row['radius']} | {row['n']} | {_fmt(row.get('avg_stability'))} | "
                f"{row['low_n']} | {_fmt(row.get('low_fraction'))} | "
                f"{row['high_n']} | {_fmt(row.get('high_fraction'))} | "
                f"{row['low_direction_lower_n']} | {row['low_direction_higher_n']} | "
                f"{row['low_direction_none_n']} | {_fmt(row.get('low_majority_share'))} |"
            )
    else:
        lines.append("No per-radius stability data was found.")
    lines.append("")

    lines.append("## Candidate Patterns")
    lines.append("")
    if result["pattern_summary"]:
        lines.append("| pattern | total | low | high |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in result["pattern_summary"]:
            lines.append(
                f"| {row['pattern']} | {row['n']} | {row['low_n']} | {row['high_n']} |"
            )
    else:
        lines.append("No pattern tags fired.")
    lines.append("")

    lines.append("## Redundancy Proxies")
    lines.append("")
    redundancy = result.get("redundancy_summary", {})
    corr = redundancy.get("correlation_with_avg_stability", {})
    rid_corr = redundancy.get("correlation_with_radius_insensitive_disagreement", {})
    if corr:
        lines.append("| proxy | corr with avg stability | corr with radius-insensitive disagreement |")
        lines.append("| --- | ---: | ---: |")
        for key in corr:
            lines.append(f"| {key} | {_fmt(corr[key])} | {_fmt(rid_corr.get(key))} |")
        lines.append("")
    group_means = redundancy.get("group_means", {})
    if group_means:
        lines.append("| group | n | total claims | base claims | addable claims | avg selected fraction |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for group, values in group_means.items():
            lines.append(
                f"| {group} | {values['n']} | {_fmt(values.get('total_claims'))} | "
                f"{_fmt(values.get('base_claims'))} | {_fmt(values.get('addable_claims'))} | "
                f"{_fmt(values.get('avg_selected_fraction'))} |"
            )
        lines.append("")

    lines.append("## Semantic Redundancy")
    lines.append("")
    semantic = result.get("semantic_redundancy_summary", {})
    if semantic.get("n_with_text"):
        lines.append(f"Rows with support text: {semantic['n_with_text']}")
        lines.append("")
        corr = semantic.get("correlation_with_avg_stability", {})
        rid_corr = semantic.get("correlation_with_radius_insensitive_disagreement", {})
        lines.append("| feature | corr with avg stability | corr with radius-insensitive disagreement |")
        lines.append("| --- | ---: | ---: |")
        for key in corr:
            lines.append(f"| {key} | {_fmt(corr[key])} | {_fmt(rid_corr.get(key))} |")
        lines.append("")
        lines.append(
            "| group | n | text count | semantic redundancy | independence | "
            "same-role pairs | duplicate pairs | tension pairs |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for group, values in semantic.get("group_means", {}).items():
            lines.append(
                f"| {group} | {values['n']} | {_fmt(values.get('semantic_text_count'))} | "
                f"{_fmt(values.get('semantic_redundancy_score'))} | "
                f"{_fmt(values.get('semantic_independence_score'))} | "
                f"{_fmt(values.get('same_role_pair_fraction'))} | "
                f"{_fmt(values.get('duplicate_pair_fraction'))} | "
                f"{_fmt(values.get('tension_pair_fraction'))} |"
            )
    else:
        lines.append("No support text directory was provided, so semantic redundancy was not computed.")
    lines.append("")

    lines.append("## Lowest Stability Examples")
    lines.append("")
    lines.extend(_examples_table(result["low_examples"]))
    lines.append("")

    lines.append("## Highest Stability Examples")
    lines.append("")
    lines.extend(_examples_table(result["high_examples"]))
    lines.append("")

    if result["source_sensitivity"]:
        lines.append("## Cross-Source Sensitivity")
        lines.append("")
        lines.append("| example | least stable | most stable | gap | pattern |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for row in result["source_sensitivity"]:
            lines.append(
                f"| {row['example_id']} | {row['least_stable_source']} | "
                f"{row['most_stable_source']} | {_fmt(row['stability_gap'])} | "
                f"{row['pattern']} |"
            )
        lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append(
        "Pattern tags are hypotheses for triage. They indicate where low-stability "
        "examples differ from high-stability examples in the sampled verdict "
        "distribution; they do not by themselves prove the underlying reasoning error."
    )
    return "\n".join(lines) + "\n"


def _examples_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| example | source | avg | first | last | majority | direction | patterns |"]
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
    for row in rows:
        patterns = ", ".join(p["pattern"] for p in row.get("patterns", [])) or "-"
        majority = (
            f"{row.get('overall_majority_label')} "
            f"({_fmt(row.get('overall_majority_share'))})"
        )
        lines.append(
            f"| {row['example_id']} | {row['source_name']} | "
            f"{_fmt(row.get('avg_stability'))} | {_fmt(row.get('first_stability'))} | "
            f"{_fmt(row.get('last_stability'))} | {majority} | "
            f"{row.get('overall_disagreement_direction') or '-'} | {patterns} |"
        )
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
