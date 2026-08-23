"""Macro-F1 vs. error-class-only F1 comparison for Experiment 2.

Answers a question raised while interpreting the Experiment 2 results: the
ARES paper's Table 1 (arXiv:2507.12948v2) reports Macro-F1 (averaged across
BOTH the sound and unsound classes), while Experiment 2's main table
(report_detection_sensitivity.py) deliberately reports error-class-only F1
per the experiment's original spec. This script recomputes Macro-F1 from our
own raw predictions -- same CV thresholds, same predicted/ground-truth sets
already used for the error-class-only numbers, just a different aggregation
formula -- so the two "ours" columns are a true apples-to-apples comparison
(identical predictions, two different metric conventions), and both sit next
to the paper's own published number for the same method/dataset/backbone.

Paper numbers below are transcribed by hand from Table 1 of
arXiv:2507.12948v2 (fetched and read directly, not from memory), CaptainCook-
Recipes column, GPT-4o-mini sub-column, F1 row. LLM-Judge is included here
for completeness of the comparison even though Experiment 2's main table
excludes it (100% real-API failure, see report_detection_sensitivity.py).

Usage:
    python -m ares_topodev.analysis.report_macro_f1_comparison \\
        --raw-results-dir ares_topodev/results \\
        --recipe-data-dir ares_topodev/vendor/ares/data/recipe_graphs \\
        --methods ares entail_prev entail_base roscoe_li_self roscoe_li_source \\
                  receval_intra receval_inter \\
        --output-dir ares_topodev/results_detection_sensitivity
"""
import argparse
import csv
import os
import statistics

from ares_topodev.analysis.detection_sensitivity import (
    bootstrap_sd_over_graphs,
    class_f1s,
    compute_ordering_detection_results,
)
from ares_topodev.analysis.report import METHOD_DISPLAY_NAMES
from ares_topodev.analysis.threshold_cv import compute_cv_thresholds, threshold_for_recipe
from ares_topodev.analysis.topodev import load_raw_results
from ares_topodev.topo_reorder.dag import load_all_recipe_dags

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Table 1, arXiv:2507.12948v2, CaptainCookRecipes / GPT-4o-mini / F1 column.
# Values are (mean, std) exactly as printed in the paper's own "mean +/- std"
# cells (std there is taken across the paper's 5 CV folds, not a bootstrap --
# a different uncertainty statistic than our bootstrap SD below, shown here
# purely as the paper's own reported number, not recomputed by us).
PAPER_MACRO_F1_CAPTAINCOOKRECIPES_GPT4OMINI = {
    "ares": (0.633, 0.010),
    "entail_prev": (0.428, 0.010),
    "entail_base": (0.589, 0.007),
    "roscoe_li_self": (0.483, 0.010),
    "roscoe_li_source": (0.361, 0.007),
    "receval_intra": (0.396, 0.010),
    "receval_inter": (0.361, 0.007),
    "llm_judge": (0.530, 0.028),
}


def _abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def run(raw_results_dir: str, recipe_data_dir: str, methods, k_folds: int, seed: int, output_dir: str):
    raw_dir = os.path.join(raw_results_dir, "raw")
    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    dags = load_all_recipe_dags(recipe_data_dir)

    rows = []
    for method in methods:
        all_results = load_raw_results(raw_dir, method)
        results = [r for r in all_results if r.get("is_complete", True)]
        if not results:
            continue

        cv = compute_cv_thresholds(results, k=k_folds, seed=seed)

        graph_mean_macro_f1 = []
        graph_mean_error_class_f1 = []
        graph_mean_sound_class_f1 = []
        for result in results:
            recipe_name = result["recipe_name"]
            dag = dags[recipe_name]
            threshold = threshold_for_recipe(cv, recipe_name)
            ordering_results = compute_ordering_detection_results(result, dag, threshold)

            universe = set(result["step_text_by_node_id"].keys())
            ground_truth = {nid for nid, label in result["ground_truth_error_by_node_id"].items() if label == 1}

            per_ordering_class_f1s = [
                class_f1s(set(o.predicted_error_set), ground_truth, universe) for o in ordering_results
            ]
            error_class_f1s = [f1_error for f1_error, _ in per_ordering_class_f1s]
            sound_class_f1s = [f1_sound for _, f1_sound in per_ordering_class_f1s]
            macro_f1s = [(e + s) / 2 for e, s in per_ordering_class_f1s]

            graph_mean_macro_f1.append(statistics.fmean(macro_f1s))
            graph_mean_error_class_f1.append(statistics.fmean(error_class_f1s))
            graph_mean_sound_class_f1.append(statistics.fmean(sound_class_f1s))

        our_macro_mean = statistics.fmean(graph_mean_macro_f1)
        our_macro_sd = bootstrap_sd_over_graphs(graph_mean_macro_f1, seed=42)
        our_error_class_mean = statistics.fmean(graph_mean_error_class_f1)
        our_error_class_sd = bootstrap_sd_over_graphs(graph_mean_error_class_f1, seed=42)
        our_sound_class_mean = statistics.fmean(graph_mean_sound_class_f1)
        our_sound_class_sd = bootstrap_sd_over_graphs(graph_mean_sound_class_f1, seed=42)

        paper_macro_f1, paper_macro_f1_sd = PAPER_MACRO_F1_CAPTAINCOOKRECIPES_GPT4OMINI.get(
            method, (None, None)
        )
        rows.append(
            {
                "method": method,
                "n_graphs": len(results),
                "paper_macro_f1": paper_macro_f1,
                "paper_macro_f1_sd": paper_macro_f1_sd,
                "our_macro_f1_mean": our_macro_mean,
                "our_macro_f1_sd": our_macro_sd,
                "our_error_class_f1_mean": our_error_class_mean,
                "our_error_class_f1_sd": our_error_class_sd,
                "our_sound_class_f1_mean": our_sound_class_mean,
                "our_sound_class_f1_sd": our_sound_class_sd,
            }
        )

    csv_path = os.path.join(aggregate_dir, "macro_f1_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = os.path.join(aggregate_dir, "macro_f1_comparison.md")
    _write_markdown(md_path, methods, rows)

    return rows


def _write_markdown(path, methods, rows):
    by_method = {r["method"]: r for r in rows}
    lines = [
        "# Macro-F1 (paper's convention) vs. Error-Class-Only F1 (Experiment 2's convention)",
        "",
        "Same predictions (same CV-thresholded raw scores) scored two ways. \"Ours, ...\" "
        "uncertainty is mean +/- bootstrap SD, resampled over graphs (see "
        "analysis/detection_sensitivity.py's bootstrap_sd_over_graphs). \"Paper's Macro-F1\" "
        "uncertainty is the paper's own reported mean +/- std across its 5 CV folds -- a "
        "different statistic, transcribed as-is, not recomputed by us.",
        "",
        "| Method | Paper's Macro-F1 | Ours, Macro-F1 | Ours, Error-Class F1 | Ours, Sound-Class F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        r = by_method.get(method)
        display = METHOD_DISPLAY_NAMES.get(method, method)
        if r is None:
            lines.append(f"| {display} | | (not yet run) | | |")
            continue
        paper = (
            f"{r['paper_macro_f1']:.3f} ± {r['paper_macro_f1_sd']:.3f}"
            if r["paper_macro_f1"] is not None
            else "n/a"
        )
        lines.append(
            f"| {display} | {paper} | {r['our_macro_f1_mean']:.3f} ± {r['our_macro_f1_sd']:.3f} | "
            f"{r['our_error_class_f1_mean']:.3f} ± {r['our_error_class_f1_sd']:.3f} | "
            f"{r['our_sound_class_f1_mean']:.3f} ± {r['our_sound_class_f1_sd']:.3f} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results-dir", default="ares_topodev/results")
    parser.add_argument("--recipe-data-dir", default="ares_topodev/vendor/ares/data/recipe_graphs")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "ares",
            "entail_prev",
            "entail_base",
            "roscoe_li_self",
            "roscoe_li_source",
            "receval_intra",
            "receval_inter",
        ],
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="ares_topodev/results_detection_sensitivity")
    args = parser.parse_args()

    rows = run(
        _abspath(args.raw_results_dir),
        _abspath(args.recipe_data_dir),
        args.methods,
        args.k_folds,
        args.seed,
        _abspath(args.output_dir),
    )
    for r in rows:
        print(
            f"{r['method']}: paper_macro_f1={r['paper_macro_f1']}, "
            f"our_macro_f1={r['our_macro_f1_mean']:.4f}, "
            f"our_error_class_f1={r['our_error_class_f1_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
