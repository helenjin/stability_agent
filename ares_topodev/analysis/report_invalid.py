"""Aggregates Dependency-Violation Sensitivity (invalid-ordering) results.

Usage:
    python -m ares_topodev.analysis.report_invalid --config ares_topodev/configs/experiment_invalid.yaml
"""
import argparse
import csv
import os

import yaml

from ares_topodev.analysis.invalid_ordering_analysis import (
    compute_case_effects,
    compute_topodev_by_node_per_recipe,
    load_invalid_raw_results,
    summarize,
)
from ares_topodev.analysis.report import METHOD_DISPLAY_NAMES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abspath(relative_to_repo_root: str) -> str:
    if os.path.isabs(relative_to_repo_root):
        return relative_to_repo_root
    return os.path.join(REPO_ROOT, relative_to_repo_root)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_invalid_report(config: dict):
    results_dir = _abspath(config["results_dir"])
    raw_dir = os.path.join(results_dir, "raw")
    aggregate_dir = os.path.join(results_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)
    baseline_raw_dir = os.path.join(_abspath(config["baseline_results_dir"]), "raw")

    all_effects = []
    summaries = []

    for method in config["methods_to_run"]:
        topodev_by_node_per_recipe = compute_topodev_by_node_per_recipe(baseline_raw_dir, method)

        all_results = load_invalid_raw_results(raw_dir, method)
        incomplete = [r["recipe_name"] for r in all_results if not r.get("is_complete", True)]
        results = [r for r in all_results if r.get("is_complete", True)]
        if incomplete:
            print(f"WARNING: {method}: skipping {len(incomplete)} incomplete recipe(s): {incomplete}")
        if not results:
            continue

        method_effects = []
        for result in results:
            topodev_by_node = topodev_by_node_per_recipe.get(result["recipe_name"])
            if topodev_by_node is None:
                print(
                    f"WARNING: {method}/{result['recipe_name']}: no baseline TopoDev data found "
                    "(baseline recipe missing/incomplete) -- skipping this recipe's cases"
                )
                continue
            method_effects.extend(compute_case_effects(result, topodev_by_node))

        if not method_effects:
            continue
        all_effects.extend(method_effects)
        summaries.append(summarize(method_effects))

    with open(os.path.join(aggregate_dir, "invalid_case_effects.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "recipe_name",
                "edge_violated",
                "violation_effect_v",
                "normalized_effect_v",
                "exceeds_noise_v",
                "correct_direction_v",
                "violation_effect_u",
                "normalized_effect_u",
                "exceeds_noise_u",
                "correct_direction_u",
            ]
        )
        for e in all_effects:
            writer.writerow(
                [
                    e.method,
                    e.recipe_name,
                    f"{e.edge_violated[0]}->{e.edge_violated[1]}",
                    e.violation_effect_v,
                    e.normalized_effect_v,
                    e.exceeds_noise_v,
                    e.correct_direction_v,
                    e.violation_effect_u,
                    e.normalized_effect_u,
                    e.exceeds_noise_u,
                    e.correct_direction_u,
                ]
            )

    with open(os.path.join(aggregate_dir, "invalid_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "n_cases",
                "frac_exceeds_noise_v",
                "frac_correct_direction_v",
                "median_normalized_effect_v",
                "frac_exceeds_noise_u",
                "frac_correct_direction_u",
                "median_normalized_effect_u",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.method,
                    s.n_cases,
                    s.frac_exceeds_noise_v,
                    s.frac_correct_direction_v,
                    s.median_normalized_effect_v,
                    s.frac_exceeds_noise_u,
                    s.frac_correct_direction_u,
                    s.median_normalized_effect_u,
                ]
            )

    _write_markdown(os.path.join(aggregate_dir, "invalid_results_table.md"), config, summaries)
    return summaries


def _write_markdown(path, config, summaries):
    lines = [
        "# Dependency-Violation Sensitivity",
        "",
        "`v` = node whose true prerequisite got displaced (claim text still names it, but it's no longer "
        "in the premise prefix). `u` = the displaced prerequisite itself.",
        "",
        "`frac_exceeds_noise` = fraction of cases where the violation moved the score by more than "
        "Experiment 1's already-measured valid-reordering range for that node -- i.e. the violation's "
        "effect stands out above ordinary harmless-reordering noise, rather than being lost in it.",
        "",
        "`frac_correct_direction` = fraction of cases where the score actually dropped (not just changed) "
        "when the dependency was violated -- the expected direction for a method that's tracking validity.",
        "",
        "| Method | N cases | Exceeds noise (v) | Correct direction (v) | Median \\|effect\\|/TopoDev (v) | "
        "Exceeds noise (u) | Correct direction (u) | Median \\|effect\\|/TopoDev (u) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by_method = {s.method: s for s in summaries}
    for label in config["methods_to_run"]:
        s = by_method.get(label)
        display = METHOD_DISPLAY_NAMES.get(label, label)
        if s is None:
            lines.append(f"| {display} | (not yet run) | | | | | | |")
        else:
            mv = f"{s.median_normalized_effect_v:.3f}" if s.median_normalized_effect_v is not None else "N/A"
            mu = f"{s.median_normalized_effect_u:.3f}" if s.median_normalized_effect_u is not None else "N/A"
            lines.append(
                f"| {display} | {s.n_cases} | {s.frac_exceeds_noise_v:.3f} | {s.frac_correct_direction_v:.3f} | "
                f"{mv} | {s.frac_exceeds_noise_u:.3f} | {s.frac_correct_direction_u:.3f} | {mu} |"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ares_topodev/configs/experiment_invalid.yaml")
    args = parser.parse_args()
    config = load_config(_abspath(args.config))
    summaries = build_invalid_report(config)
    for s in summaries:
        print(
            f"{s.method}: frac_exceeds_noise(v)={s.frac_exceeds_noise_v:.3f}, "
            f"frac_correct_direction(v)={s.frac_correct_direction_v:.3f}, n={s.n_cases}"
        )


if __name__ == "__main__":
    main()
