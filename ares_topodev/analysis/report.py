"""Aggregates raw per-ordering results (written by eval_harness/run_experiment.py)
into the TopoDev summary tables and the main results table.

Only ever reads what's actually on disk under `results/raw/<method>/*.json` --
never invents a row for a method/dataset that hasn't been run.

Usage:
    python -m ares_topodev.analysis.report --config ares_topodev/configs/experiment.yaml
"""
import argparse
import csv
import os
import sys

import yaml

from ares_topodev.analysis.diagnostics import (
    write_final_conclusion_deviation_csv,
    write_num_valid_orderings_csv,
    write_per_step_deviation_csv,
    write_position_vs_score_csv,
)
from ares_topodev.analysis.error_recovery import choose_threshold, compute_error_recovery
from ares_topodev.analysis.topodev import (
    compute_per_example_topodev,
    compute_per_example_topovar,
    load_raw_results,
    summarize_topodev,
    summarize_topovar,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHOD_DISPLAY_NAMES = {
    "ares": "ARES",
    "entail_prev": "Entail-Prev",
    "entail_base": "Entail-Base",
    "roscoe_li_self": "ROSCOE-LI-Self",
    "roscoe_li_source": "ROSCOE-LI-Source",
    "receval_intra": "ReCEval-Intra",
    "receval_inter": "ReCEval-Inter",
    "llm_judge": "LLM-Judge",
}


def _abspath(relative_to_repo_root: str) -> str:
    if os.path.isabs(relative_to_repo_root):
        return relative_to_repo_root
    return os.path.join(REPO_ROOT, relative_to_repo_root)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_report(config: dict):
    results_dir = _abspath(config["results_dir"])
    raw_dir = os.path.join(results_dir, "raw")
    aggregate_dir = os.path.join(results_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    raw_results_by_method = {}
    per_example_by_method = {}
    summaries = []
    topovar_summaries = []

    per_example_rows = []
    topovar_rows = []
    error_recovery_rows = []
    error_recovery_thresholds = {}

    for method in config["methods_to_run"]:
        all_results = load_raw_results(raw_dir, method)
        incomplete = [r["recipe_name"] for r in all_results if not r.get("is_complete", True)]
        if incomplete:
            print(
                f"WARNING: {method}: skipping {len(incomplete)} incomplete recipe(s) from aggregation "
                f"(still mid-run or interrupted): {incomplete}",
                file=sys.stderr,
            )
        raw_results = [r for r in all_results if r.get("is_complete", True)]
        raw_results_by_method[method] = raw_results
        if not raw_results:
            continue
        per_example = [compute_per_example_topodev(r) for r in raw_results]
        per_example_by_method[method] = per_example

        summary = summarize_topodev(per_example)
        summaries.append(summary)

        for pe in per_example:
            per_example_rows.append(pe)

        per_example_var = [compute_per_example_topovar(r) for r in raw_results]
        topovar_summaries.append(summarize_topovar(per_example_var))
        for pev in per_example_var:
            topovar_rows.append(pev)

        write_per_step_deviation_csv(
            os.path.join(aggregate_dir, "per_step_deviation.csv"), method, raw_results, per_example
        )
        write_final_conclusion_deviation_csv(
            os.path.join(aggregate_dir, "final_conclusion_deviation.csv"), method, per_example
        )
        write_position_vs_score_csv(os.path.join(aggregate_dir, "position_vs_score.csv"), method, raw_results)

        threshold = choose_threshold(raw_results)
        error_recovery_thresholds[method] = threshold
        for r in raw_results:
            error_recovery_rows.append(compute_error_recovery(r, threshold))

    write_num_valid_orderings_csv(os.path.join(aggregate_dir, "num_valid_orderings.csv"), raw_results_by_method)

    with open(os.path.join(aggregate_dir, "topodev_per_example.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "recipe_name", "topodev", "num_orderings_used"])
        for pe in per_example_rows:
            writer.writerow([pe.dataset, pe.method, pe.recipe_name, pe.topodev, pe.num_orderings_used])

    with open(os.path.join(aggregate_dir, "topodev_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "n_examples", "mean", "median", "std", "ci_low_95", "ci_high_95"])
        for s in summaries:
            writer.writerow([s.dataset, s.method, s.n_examples, s.mean, s.median, s.std, s.ci_low, s.ci_high])

    with open(os.path.join(aggregate_dir, "topovar_per_example.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "recipe_name", "topovar", "num_orderings_used"])
        for pev in topovar_rows:
            writer.writerow([pev.dataset, pev.method, pev.recipe_name, pev.topovar, pev.num_orderings_used])

    with open(os.path.join(aggregate_dir, "topovar_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "n_examples", "mean", "median", "std", "ci_low_95", "ci_high_95"])
        for s in topovar_summaries:
            writer.writerow([s.dataset, s.method, s.n_examples, s.mean, s.median, s.std, s.ci_low, s.ci_high])

    with open(os.path.join(aggregate_dir, "error_recovery_per_example.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "method",
                "recipe_name",
                "threshold",
                "num_orderings",
                "ground_truth_error_node_ids",
                "mean_pairwise_jaccard",
                "frac_orderings_exact_match_gt",
                "mean_recall",
                "mean_precision",
            ]
        )
        for er in error_recovery_rows:
            writer.writerow(
                [
                    er.dataset,
                    er.method,
                    er.recipe_name,
                    er.threshold,
                    er.num_orderings,
                    ";".join(er.ground_truth_error_node_ids),
                    er.mean_pairwise_jaccard,
                    er.frac_orderings_exact_match_gt,
                    er.mean_recall,
                    er.mean_precision,
                ]
            )

    error_recovery_summary_by_method = _summarize_error_recovery(error_recovery_rows, error_recovery_thresholds)
    with open(os.path.join(aggregate_dir, "error_recovery_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "threshold",
                "n_examples",
                "mean_pairwise_jaccard",
                "mean_frac_orderings_exact_match_gt",
                "mean_recall",
                "mean_precision",
            ]
        )
        for method, row in error_recovery_summary_by_method.items():
            writer.writerow(
                [
                    method,
                    row["threshold"],
                    row["n_examples"],
                    row["mean_pairwise_jaccard"],
                    row["mean_frac_orderings_exact_match_gt"],
                    row["mean_recall"],
                    row["mean_precision"],
                ]
            )

    _write_main_results_table(
        os.path.join(aggregate_dir, "main_results_table.md"),
        config,
        summaries,
        topovar_summaries,
        error_recovery_summary_by_method,
    )
    return summaries, topovar_summaries, error_recovery_summary_by_method


def _summarize_error_recovery(rows, thresholds_by_method) -> dict:
    by_method = {}
    for er in rows:
        by_method.setdefault(er.method, []).append(er)
    summary = {}
    for method, ers in by_method.items():
        n = len(ers)
        summary[method] = {
            "threshold": thresholds_by_method[method],
            "n_examples": n,
            "mean_pairwise_jaccard": sum(e.mean_pairwise_jaccard for e in ers) / n,
            "mean_frac_orderings_exact_match_gt": sum(e.frac_orderings_exact_match_gt for e in ers) / n,
            "mean_recall": sum(e.mean_recall for e in ers) / n,
            "mean_precision": sum(e.mean_precision for e in ers) / n,
        }
    return summary


def _write_main_results_table(path: str, config: dict, summaries, topovar_summaries, error_recovery_summary_by_method):
    lines = [
        "# Main Results Table",
        "",
        "TopoDev = mean max-min range per step across sampled orderings (sensitive to a single outlier ordering).",
        "TopoVar = mean per-step sample variance across sampled orderings (more robust to a single outlier, but on a squared-score scale).",
        "",
        "| Dataset | Method | TopoDev (mean) | TopoDev 95% CI | TopoVar (mean) | TopoVar 95% CI | N |",
        "|---|---|---|---|---|---|---|",
    ]

    for label in ["ares", "entail_prev", "entail_base"]:
        display = METHOD_DISPLAY_NAMES.get(label, label)
        lines.append(
            f"| ClaimTrees | {display} | N/A | N/A | N/A | N/A | 0 | "
            f"-- every released ClaimTrees config is a strict linear chain "
            f"(exactly one valid topological order); TopoDev/TopoVar are "
            f"structurally undefined, not merely unmeasured. See VENDOR.md."
        )

    by_method = {s.method: s for s in summaries}
    var_by_method = {s.method: s for s in topovar_summaries}
    for label in config["methods_to_run"]:
        s = by_method.get(label)
        v = var_by_method.get(label)
        display = METHOD_DISPLAY_NAMES.get(label, label)
        if s is None:
            lines.append(f"| CaptainCookRecipes | {display} | (not yet run) | | | | 0 |")
        else:
            lines.append(
                f"| CaptainCookRecipes | {display} | {s.mean:.4f} | [{s.ci_low:.4f}, {s.ci_high:.4f}] | "
                f"{v.mean:.4f} | [{v.ci_low:.4f}, {v.ci_high:.4f}] | {s.n_examples} |"
            )

    lines += [
        "",
        "## Error-Recovery Consistency",
        "",
        "Does a method flag the *same* ground-truth error node(s) as errors regardless of which valid "
        "ordering presented them? Threshold per method chosen to maximize pooled Macro-F1 against ground "
        "truth (see analysis/error_recovery.py) -- a simplification of the paper's cross-validated procedure, "
        "not a reproduction of it. `jaccard`=1.0 means the predicted-error node set never changes across "
        "orderings for a recipe; `exact_match_gt`=1.0 means every ordering's predicted set exactly equals "
        "the true error set.",
        "",
        "| Method | Threshold | Mean pairwise Jaccard (predicted-set stability) | "
        "Frac. orderings exactly matching ground truth | Mean recall | Mean precision | N |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in config["methods_to_run"]:
        row = error_recovery_summary_by_method.get(label)
        display = METHOD_DISPLAY_NAMES.get(label, label)
        if row is None:
            lines.append(f"| {display} | (not yet run) | | | | | 0 |")
        else:
            lines.append(
                f"| {display} | {row['threshold']:.4f} | {row['mean_pairwise_jaccard']:.4f} | "
                f"{row['mean_frac_orderings_exact_match_gt']:.4f} | {row['mean_recall']:.4f} | "
                f"{row['mean_precision']:.4f} | {row['n_examples']} |"
            )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ares_topodev/configs/experiment.yaml")
    args = parser.parse_args()
    config = load_config(_abspath(args.config))
    summaries, topovar_summaries, error_recovery_summary_by_method = build_report(config)
    var_by_method = {s.method: s for s in topovar_summaries}
    for s in summaries:
        v = var_by_method.get(s.method)
        er = error_recovery_summary_by_method.get(s.method, {})
        print(
            f"{s.dataset} / {s.method}: mean TopoDev = {s.mean:.4f}, mean TopoVar = {v.mean:.4f}, "
            f"mean predicted-set Jaccard = {er.get('mean_pairwise_jaccard', float('nan')):.4f} (n={s.n_examples})"
        )


if __name__ == "__main__":
    main()
