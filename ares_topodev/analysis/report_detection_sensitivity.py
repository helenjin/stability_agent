"""Experiment 2: error-detection sensitivity under valid reordering.

Reuses Experiment 1's already-computed, already-verified valid orderings and
scores directly -- no new model inference for ARES/Entail-Prev/Entail-Base.
See analysis/detection_sensitivity.py and analysis/threshold_cv.py for the
metric definitions and the 5-fold cross-validated threshold procedure.

Usage:
    python -m ares_topodev.analysis.report_detection_sensitivity \\
        --raw-results-dir ares_topodev/results \\
        --recipe-data-dir ares_topodev/vendor/ares/data/recipe_graphs \\
        --methods ares entail_prev entail_base \\
        --k-folds 5 --seed 42 \\
        --output-dir ares_topodev/results_detection_sensitivity
"""
import argparse
import csv
import os

from ares_topodev.analysis.detection_sensitivity import (
    compute_graph_sensitivity,
    summarize_across_graphs,
)
from ares_topodev.analysis.report import METHOD_DISPLAY_NAMES
from ares_topodev.analysis.threshold_cv import compute_cv_thresholds
from ares_topodev.analysis.topodev import load_raw_results
from ares_topodev.topo_reorder.dag import load_all_recipe_dags

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def run(raw_results_dir: str, recipe_data_dir: str, methods, k_folds: int, seed: int, output_dir: str):
    raw_dir = os.path.join(raw_results_dir, "raw")
    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    dags = load_all_recipe_dags(recipe_data_dir)

    long_form_rows = []
    order_level_rows = []
    graph_summaries_all = []
    cross_graph_summaries = []
    skipped_incomplete_by_method = {}

    for method in methods:
        all_results = load_raw_results(raw_dir, method)
        incomplete = [r["recipe_name"] for r in all_results if not r.get("is_complete", True)]
        skipped_incomplete_by_method[method] = incomplete
        results = [r for r in all_results if r.get("is_complete", True)]
        if not results:
            continue

        cv = compute_cv_thresholds(results, k=k_folds, seed=seed)

        graph_summaries = []
        for result in results:
            recipe_name = result["recipe_name"]
            dag = dags[recipe_name]
            summary, ordering_results = compute_graph_sensitivity(result, dag, cv)
            graph_summaries.append(summary)

            # canonical "original position" reference = ordering_index 0
            original_order = result["orderings"][0]["topo_order_step_ids"]
            original_position = {str(nid): i for i, nid in enumerate(original_order)}

            for ordering, ores in zip(result["orderings"], ordering_results):
                order_level_rows.append(
                    {
                        "dataset": result["dataset"],
                        "recipe_name": recipe_name,
                        "ordering_index": ordering["ordering_index"],
                        "method": method,
                        "threshold": summary.threshold,
                        "fold": summary.fold,
                        "precision": ores.precision,
                        "recall": ores.recall,
                        "f1": ores.f1,
                        "precision_zero_division": ores.precision_zero_division,
                        "recall_zero_division": ores.recall_zero_division,
                        "predicted_error_set": ";".join(ores.predicted_error_set),
                        "ground_truth_error_set": ";".join(summary.ground_truth_error_set),
                    }
                )

                reordered_position = {str(nid): i for i, nid in enumerate(ordering["topo_order_step_ids"])}
                gt = set(summary.ground_truth_error_set)
                predicted = set(ores.predicted_error_set)
                for nid, text in result["step_text_by_node_id"].items():
                    score = ordering["scores_by_node_id"].get(nid)
                    if score is None:
                        continue
                    long_form_rows.append(
                        {
                            "dataset": result["dataset"],
                            "recipe_name": recipe_name,
                            "ordering_index": ordering["ordering_index"],
                            "claim_id": nid,
                            "original_position": original_position.get(nid),
                            "reordered_position": reordered_position.get(nid),
                            "claim_text": text,
                            "ground_truth_error_label": 1 if nid in gt else 0,
                            "method": method,
                            "raw_score": score,
                            "predicted_error_label": 1 if nid in predicted else 0,
                            "model": result["model"],
                            "threshold": summary.threshold,
                            "hyperparams": result.get("hyperparams", {}),
                            "seed": result["seed"],
                        }
                    )

        graph_summaries_all.extend(graph_summaries)
        cross_graph_summaries.append(summarize_across_graphs(graph_summaries))

    # --- write outputs ---
    _write_csv(
        os.path.join(aggregate_dir, "long_form_results.csv"),
        long_form_rows,
        [
            "dataset",
            "recipe_name",
            "ordering_index",
            "claim_id",
            "original_position",
            "reordered_position",
            "claim_text",
            "ground_truth_error_label",
            "method",
            "raw_score",
            "predicted_error_label",
            "model",
            "threshold",
            "hyperparams",
            "seed",
        ],
    )
    _write_csv(
        os.path.join(aggregate_dir, "order_level_summary.csv"),
        order_level_rows,
        [
            "dataset",
            "recipe_name",
            "ordering_index",
            "method",
            "threshold",
            "fold",
            "precision",
            "recall",
            "f1",
            "precision_zero_division",
            "recall_zero_division",
            "predicted_error_set",
            "ground_truth_error_set",
        ],
    )
    _write_csv(
        os.path.join(aggregate_dir, "graph_level_sensitivity.csv"),
        [
            {
                "method": g.method,
                "recipe_name": g.recipe_name,
                "fold": g.fold,
                "threshold": g.threshold,
                "num_orderings": g.num_orderings,
                "mean_f1": g.mean_f1,
                "min_f1": g.min_f1,
                "max_f1": g.max_f1,
                "delta_f1": g.delta_f1,
                "mean_pairwise_jaccard": g.mean_pairwise_jaccard,
                "exact_match_rate": g.exact_match_rate,
                "flip_rate": g.flip_rate,
                "num_zero_predicted_positive_orderings": g.num_zero_predicted_positive_orderings,
            }
            for g in graph_summaries_all
        ],
        [
            "method",
            "recipe_name",
            "fold",
            "threshold",
            "num_orderings",
            "mean_f1",
            "min_f1",
            "max_f1",
            "delta_f1",
            "mean_pairwise_jaccard",
            "exact_match_rate",
            "flip_rate",
            "num_zero_predicted_positive_orderings",
        ],
    )
    _write_csv(
        os.path.join(aggregate_dir, "cross_graph_summary.csv"),
        [
            {
                "method": s.method,
                "n_graphs": s.n_graphs,
                "mean_f1_mean": s.mean_f1_mean,
                "mean_f1_ci_low": s.mean_f1_ci[0],
                "mean_f1_ci_high": s.mean_f1_ci[1],
                "delta_f1_mean": s.delta_f1_mean,
                "delta_f1_median": s.delta_f1_median,
                "delta_f1_ci_low": s.delta_f1_ci[0],
                "delta_f1_ci_high": s.delta_f1_ci[1],
                "mean_jaccard": s.mean_jaccard,
                "mean_exact_match_rate": s.mean_exact_match_rate,
                "mean_flip_rate": s.mean_flip_rate,
                "total_zero_predicted_positive_orderings": s.total_zero_predicted_positive_orderings,
            }
            for s in cross_graph_summaries
        ],
        [
            "method",
            "n_graphs",
            "mean_f1_mean",
            "mean_f1_ci_low",
            "mean_f1_ci_high",
            "delta_f1_mean",
            "delta_f1_median",
            "delta_f1_ci_low",
            "delta_f1_ci_high",
            "mean_jaccard",
            "mean_exact_match_rate",
            "mean_flip_rate",
            "total_zero_predicted_positive_orderings",
        ],
    )

    _write_main_table(os.path.join(aggregate_dir, "main_results_table.md"), methods, cross_graph_summaries)
    _write_secondary_table(os.path.join(aggregate_dir, "secondary_results_table.md"), methods, cross_graph_summaries)

    for method, incomplete in skipped_incomplete_by_method.items():
        if incomplete:
            print(f"WARNING: {method}: skipped {len(incomplete)} incomplete recipe(s): {incomplete}")

    total_zero_pred = sum(s.total_zero_predicted_positive_orderings for s in cross_graph_summaries)
    print(f"Zero-predicted-positive orderings across all methods: {total_zero_pred} (precision set to 0.0 by convention)")

    return cross_graph_summaries


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_main_table(path, methods, summaries):
    by_method = {s.method: s for s in summaries}
    lines = [
        "# Experiment 2 Main Table: Error-Detection Sensitivity Under Valid Reordering",
        "",
        "| Method | Mean F1 ↑ | ΔF1 ↓ | N graphs |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        s = by_method.get(method)
        display = METHOD_DISPLAY_NAMES.get(method, method)
        if s is None:
            lines.append(f"| {display} | (not yet run) | | |")
        else:
            lines.append(f"| {display} | {s.mean_f1_mean:.4f} | {s.delta_f1_mean:.4f} | {s.n_graphs} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_secondary_table(path, methods, summaries):
    by_method = {s.method: s for s in summaries}
    lines = [
        "# Experiment 2 Secondary Table: Error-Set Consistency",
        "",
        "| Method | Error-Set Jaccard ↑ | Exact-Match ↑ | Flip Rate ↓ |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        s = by_method.get(method)
        display = METHOD_DISPLAY_NAMES.get(method, method)
        if s is None:
            lines.append(f"| {display} | (not yet run) | | |")
        else:
            lines.append(f"| {display} | {s.mean_jaccard:.4f} | {s.mean_exact_match_rate:.4f} | {s.mean_flip_rate:.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results-dir", default="ares_topodev/results")
    parser.add_argument("--recipe-data-dir", default="ares_topodev/vendor/ares/data/recipe_graphs")
    parser.add_argument("--methods", nargs="+", default=["ares", "entail_prev", "entail_base"])
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="ares_topodev/results_detection_sensitivity")
    args = parser.parse_args()

    summaries = run(
        _abspath(args.raw_results_dir),
        _abspath(args.recipe_data_dir),
        args.methods,
        args.k_folds,
        args.seed,
        _abspath(args.output_dir),
    )
    for s in summaries:
        print(
            f"{s.method}: mean_F1={s.mean_f1_mean:.4f} {s.mean_f1_ci}, "
            f"delta_F1={s.delta_f1_mean:.4f} {s.delta_f1_ci}, n={s.n_graphs}"
        )


if __name__ == "__main__":
    main()
