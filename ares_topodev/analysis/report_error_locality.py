"""Error-detection breakdown by causal relationship to the injected error,
not just overall F1 (see analysis/detection_sensitivity.py for that).

For each derived node, `recipe_example.build_recipe_example`'s
`error_category_by_node_id` classifies it as:
  source              directly needs the deleted ingredient
  propagated          forward-reachable from a source (a "downstream" error)
  ancestor_of_source  causally upstream of a source -- ground-truth-sound
  independent         ground-truth-sound, causally unrelated to the error

Source/propagated nodes are ground-truth errors, so only RECALL is
meaningful there (precision is trivially 1.0 by construction -- these
categories contain no ground-truth-sound nodes to false-positive against).
Ancestor/independent nodes are ground-truth-sound, so only the FALSE
POSITIVE RATE is meaningful (precision is trivially 0.0 the same way).

This directly answers two of the section-15 framing questions that plain
error-class F1 doesn't distinguish: does a method's F1 advantage come from
catching source errors, catching propagated errors, or from an unrelated
change in overall confidence? And does graph-local evaluation reduce false
positives specifically on causally-independent branches, as the naive
hypothesis would predict (checked directly against real results, not
assumed)?

Reuses `analysis.threshold_cv.compute_cv_thresholds` (same recipe-level CV
thresholds Experiment 2 uses) and `analysis.detection_sensitivity.predicted_error_set`
unmodified -- only the per-node category grouping is new here.

Usage:
    python -m ares_topodev.analysis.report_error_locality \\
        --raw-results-dir ares_topodev/results_sager \\
        --recipe-data-dir ares_topodev/vendor/ares/data/recipe_graphs \\
        --methods ares sager \\
        --k-folds 5 --seed 42 \\
        --output-dir ares_topodev/results_sager
"""
import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from ares_topodev.analysis.detection_sensitivity import predicted_error_set
from ares_topodev.analysis.threshold_cv import compute_cv_thresholds, threshold_for_recipe
from ares_topodev.analysis.topodev import load_raw_results
from ares_topodev.eval_harness.recipe_example import build_recipe_example
from ares_topodev.topo_reorder.dag import extract_recipe_dag, load_recipe_json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATEGORIES = ("source", "propagated", "ancestor_of_source", "independent")


def _abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def category_by_node_id(recipe_name: str, recipe_data_dir: str, base_seed: int = 42) -> Dict[str, str]:
    """String-keyed (to match raw-result node-id keys) error category per node."""
    raw = load_recipe_json(os.path.join(recipe_data_dir, f"{recipe_name}.json"))
    dag = extract_recipe_dag(raw, recipe_name)
    example = build_recipe_example(dag, raw, base_seed=base_seed)
    return {str(nid): category for nid, category in example.error_category_by_node_id.items()}


@dataclass
class CategoryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def fp_rate(self):
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else None


def run(raw_results_dir: str, recipe_data_dir: str, methods: List[str], k_folds: int, seed: int, output_dir: str):
    raw_dir = os.path.join(raw_results_dir, "raw")
    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    rows = []
    for method in methods:
        results = [r for r in load_raw_results(raw_dir, method) if r.get("is_complete", True)]
        if not results:
            continue
        cv = compute_cv_thresholds(results, k=k_folds, seed=seed)

        counts_by_category: Dict[str, CategoryCounts] = {c: CategoryCounts() for c in CATEGORIES}
        for result in results:
            recipe_name = result["recipe_name"]
            categories = category_by_node_id(recipe_name, recipe_data_dir)
            threshold = threshold_for_recipe(cv, recipe_name)
            ground_truth = {nid for nid, label in result["ground_truth_error_by_node_id"].items() if label == 1}

            for ordering in result["orderings"]:
                predicted = predicted_error_set(ordering["scores_by_node_id"], threshold)
                for nid in ordering["scores_by_node_id"]:
                    category = categories.get(nid)
                    if category is None:
                        continue
                    is_gt_error = nid in ground_truth
                    is_pred_error = nid in predicted
                    c = counts_by_category[category]
                    if is_gt_error and is_pred_error:
                        c.tp += 1
                    elif is_gt_error and not is_pred_error:
                        c.fn += 1
                    elif not is_gt_error and is_pred_error:
                        c.fp += 1
                    else:
                        c.tn += 1

        for category in CATEGORIES:
            c = counts_by_category[category]
            rows.append(
                {
                    "method": method,
                    "category": category,
                    "n": c.n,
                    "recall": c.recall,
                    "fp_rate": c.fp_rate,
                    "tp": c.tp,
                    "fp": c.fp,
                    "fn": c.fn,
                    "tn": c.tn,
                }
            )

    with open(os.path.join(aggregate_dir, "error_locality_breakdown.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "category", "n", "recall", "fp_rate", "tp", "fp", "fn", "tn"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _write_markdown_table(os.path.join(aggregate_dir, "error_locality_breakdown.md"), rows, methods)
    return rows


def _write_markdown_table(path: str, rows: List[dict], methods: List[str]):
    by_method_category = {(r["method"], r["category"]): r for r in rows}
    lines = [
        "# Error-detection breakdown by causal relationship to the injected error",
        "",
        "Source/propagated rows: only recall is meaningful (these categories are all "
        "ground-truth errors -- precision is trivially 1.0). Ancestor/independent rows: "
        "only false-positive rate is meaningful (these categories are all ground-truth "
        "sound -- precision is trivially 0.0).",
        "",
        "| Method | Category | n | Recall | FP rate |",
        "|---|---|---:|---:|---:|",
    ]
    for method in methods:
        for category in CATEGORIES:
            r = by_method_category.get((method, category))
            if r is None:
                continue
            recall = f"{r['recall']:.4f}" if r["recall"] is not None else "--"
            fp_rate = f"{r['fp_rate']:.4f}" if r["fp_rate"] is not None else "--"
            lines.append(f"| {method} | {category} | {r['n']} | {recall} | {fp_rate} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results-dir", default="ares_topodev/results_sager")
    parser.add_argument("--recipe-data-dir", default="ares_topodev/vendor/ares/data/recipe_graphs")
    parser.add_argument("--methods", nargs="+", default=["ares", "sager"])
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="ares_topodev/results_sager")
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
        recall = f"{r['recall']:.4f}" if r["recall"] is not None else "n/a"
        fp_rate = f"{r['fp_rate']:.4f}" if r["fp_rate"] is not None else "n/a"
        print(f"{r['method']:8s} {r['category']:20s} n={r['n']:5d} recall={recall} fp_rate={fp_rate}")


if __name__ == "__main__":
    main()
