"""Diagnostic analyses beyond the headline TopoDev number:
  - per-step deviation (which reasoning steps are most order-sensitive)
  - final-conclusion deviation (already computed per-example in topodev.py;
    collected here into one table)
  - score vs. sequence position (does a step get systematically different
    scores depending on where in the chain it happens to land?)
  - number of valid topological orderings sampled/estimated per example
"""
import csv
import os
from typing import List

from ares_topodev.analysis.topodev import PerExampleTopoDev


def write_per_step_deviation_csv(path: str, method: str, raw_results: List[dict], per_example: List[PerExampleTopoDev]):
    by_recipe = {pe.recipe_name: pe for pe in per_example}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "recipe_name", "node_id", "step_text", "ground_truth_error", "max_minus_min"])
        for result in raw_results:
            pe = by_recipe.get(result["recipe_name"])
            if pe is None:
                continue
            for node_id, deviation in pe.per_step_deviation.items():
                writer.writerow(
                    [
                        result["dataset"],
                        method,
                        result["recipe_name"],
                        node_id,
                        result["step_text_by_node_id"].get(node_id, ""),
                        result["ground_truth_error_by_node_id"].get(node_id, ""),
                        deviation,
                    ]
                )


def write_final_conclusion_deviation_csv(path: str, method: str, per_example: List[PerExampleTopoDev]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "recipe_name", "final_node_id", "final_conclusion_deviation"])
        for pe in per_example:
            writer.writerow([pe.dataset, method, pe.recipe_name, pe.final_node_id, pe.final_conclusion_deviation])


def write_position_vs_score_csv(path: str, method: str, raw_results: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "recipe_name", "ordering_index", "node_id", "position", "score"])
        for result in raw_results:
            for ordering in result["orderings"]:
                order = ordering["topo_order_step_ids"]
                for position, node_id in enumerate(order):
                    score = ordering["scores_by_node_id"].get(str(node_id))
                    if score is None:
                        continue
                    writer.writerow(
                        [result["dataset"], method, result["recipe_name"], ordering["ordering_index"], node_id, position, score]
                    )


def write_num_valid_orderings_csv(path: str, raw_results_by_method: dict):
    """raw_results_by_method: {method: [result_dict, ...]}. Since
    num_valid_orderings_estimate is a property of the DAG (not the method), we
    just take it from whichever method's results are available for a recipe,
    documented once per recipe rather than once per (recipe, method)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = {}
    for method, results in raw_results_by_method.items():
        for result in results:
            seen.setdefault(
                result["recipe_name"],
                {
                    "recipe_name": result["recipe_name"],
                    "num_orderings_used": result["num_orderings_used"],
                    "num_orderings_requested": result["num_orderings_requested"],
                    "topo_sampling_exhausted": result["topo_sampling_exhausted"],
                    "num_valid_orderings_estimate": result["num_valid_orderings_estimate"]["value"],
                    "estimate_is_exact": result["num_valid_orderings_estimate"]["exact"],
                },
            )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "recipe_name",
                "num_orderings_used",
                "num_orderings_requested",
                "topo_sampling_exhausted",
                "num_valid_orderings_estimate",
                "estimate_is_exact",
            ]
        )
        for row in seen.values():
            writer.writerow(
                [
                    row["recipe_name"],
                    row["num_orderings_used"],
                    row["num_orderings_requested"],
                    row["topo_sampling_exhausted"],
                    row["num_valid_orderings_estimate"],
                    row["estimate_is_exact"],
                ]
            )
