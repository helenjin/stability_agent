"""Experiment 2: evaluate methods on minimal, single-edge dependency-violating
orderings, and compare against Experiment 1's already-computed valid-ordering
baseline for the exact same (recipe, node).

Reuses the *first* valid ordering Experiment 1 sampled (same seed -> same
draw, see topo_reorder/topo_sample.py) as the base ordering to swap from, so
that ordering's scores don't need to be recomputed -- we read them straight
out of Experiment 1's saved `orderings[0]` for a free, exact-match baseline.
Only the swapped (invalid) orderings need fresh API calls.

Usage:
    python -m ares_topodev.eval_harness.run_invalid_experiment \\
        --config ares_topodev/configs/experiment_invalid.yaml [--limit N] [--dry-run]
"""
import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time

import yaml

from ares_topodev.eval_harness import _bootstrap  # noqa: F401
from ares_topodev.eval_harness.recipe_example import apply_ordering, build_recipe_example
from ares_topodev.eval_harness.run_experiment import (
    _abspath,
    build_data_entry,
    build_entailment_model,
    build_scorers,
    load_config,
)
from ares_topodev.topo_reorder.dag import extract_recipe_dag, load_all_recipe_dags, load_recipe_json
from ares_topodev.topo_reorder.invalid_topo import generate_invalid_orderings
from ares_topodev.topo_reorder.topo_sample import sample_orderings

_print_lock = None


def _log(message: str):
    if _print_lock is not None:
        with _print_lock:
            print(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr)


def load_baseline_ordering0(baseline_raw_dir: str, method: str, recipe_name: str):
    """Reads Experiment 1's ordering_index==0 scores for one (method, recipe)
    -- the exact valid-ordering baseline this experiment's swaps are derived
    from. Returns None if not available (e.g. baseline run hasn't reached
    this recipe yet), so callers can skip gracefully rather than block."""
    path = os.path.join(baseline_raw_dir, method, f"{recipe_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        result = json.load(f)
    ordering0 = next((o for o in result["orderings"] if o["ordering_index"] == 0), None)
    if ordering0 is None:
        return None
    return {
        "topo_order_step_ids": ordering0["topo_order_step_ids"],
        "scores_by_node_id": ordering0["scores_by_node_id"],
    }


def process_recipe_invalid(
    recipe_name, dag, data_dir, config, scorers, resolved_kwargs_by_label, cached_llm, raw_dir, baseline_raw_dir, dry_run
):
    t0 = time.time()
    raw = load_recipe_json(os.path.join(data_dir, f"{recipe_name}.json"))
    example = build_recipe_example(
        dag, raw, base_seed=config["seed"], raw_claims_shuffle_idx=config["raw_claims_shuffle_idx"]
    )

    # Same seed as Experiment 1 -> identical first draw -> identical ordering.
    base_ordering = sample_orderings(dag, k=1, seed=config["seed"], max_attempts=config["max_topo_attempts"]).orderings[0]
    cases = generate_invalid_orderings(dag, base_ordering, max_cases=config.get("max_cases_per_recipe"))

    baseline_by_method = {
        label: load_baseline_ordering0(baseline_raw_dir, label, recipe_name) for label in config["methods_to_run"]
    }
    for label, baseline in baseline_by_method.items():
        if baseline is not None and baseline["topo_order_step_ids"] != base_ordering:
            _log(
                f"WARNING [{recipe_name}/{label}]: baseline ordering_index==0 does not match this run's "
                "base_ordering -- baseline config/seed may differ from this experiment's; comparison will "
                "still be computed but isn't a clean like-for-like match."
            )

    per_method_cases = {label: [] for label in config["methods_to_run"]}

    for case_index, case in enumerate(cases):
        derived_claims = apply_ordering(example, case.ordering)
        data_entry = build_data_entry(example.raw_claims, derived_claims)

        for label, scorer in scorers.items():
            result = scorer.get_stability_rate(data_entry)
            scores_by_node_id = {
                str(node_id): score for node_id, score in zip(case.ordering, result.stability_rates)
            }
            u, v = case.edge_violated
            baseline = baseline_by_method[label]
            per_method_cases[label].append(
                {
                    "case_index": case_index,
                    "edge_violated": [u, v],
                    "invalid_ordering_step_ids": case.ordering,
                    "scores_by_node_id": scores_by_node_id,
                    "baseline_valid_score_u": baseline["scores_by_node_id"].get(str(u)) if baseline else None,
                    "baseline_valid_score_v": baseline["scores_by_node_id"].get(str(v)) if baseline else None,
                    "invalid_score_u": scores_by_node_id.get(str(u)),
                    "invalid_score_v": scores_by_node_id.get(str(v)),
                }
            )

        out_by_label = {}
        for label in config["methods_to_run"]:
            out_by_label[label] = {
                "dataset": "captaincookrecipes",
                "recipe_name": recipe_name,
                "method": label,
                "model": "mock-llm" if dry_run else config["backbone_model"],
                "seed": config["seed"],
                "base_ordering_step_ids": base_ordering,
                "num_cases_total": len(cases),
                "num_cases_done": len(per_method_cases[label]),
                "is_complete": case_index == len(cases) - 1,
                "ground_truth_error_by_node_id": {
                    str(k): v for k, v in example.ground_truth_error_by_node_id.items()
                },
                "cases": per_method_cases[label],
            }
            out_path = os.path.join(raw_dir, label, f"{recipe_name}.json")
            tmp_path = out_path + f".tmp.{threading.get_ident()}"
            with open(tmp_path, "w") as f:
                json.dump(out_by_label[label], f, indent=2)
            os.replace(tmp_path, out_path)

        _log(f"[{recipe_name}] invalid case {case_index + 1}/{len(cases)} saved (edge={case.edge_violated})")

    dt = time.time() - t0
    _log(f"[{recipe_name}] {len(cases)} invalid cases done, time={dt:.1f}s")


def run(config: dict, limit=None, dry_run=False):
    global _print_lock

    data_dir = _abspath(config["recipe_data_dir"])
    results_dir = _abspath(config["results_dir"])
    raw_dir = os.path.join(results_dir, "raw")
    baseline_raw_dir = os.path.join(_abspath(config["baseline_results_dir"]), "raw")

    dags = load_all_recipe_dags(data_dir)
    recipe_names = sorted(dags.keys())
    if limit is not None:
        recipe_names = recipe_names[:limit]

    entailment_model, cached_llm = build_entailment_model(config, dry_run=dry_run)
    scorers, resolved_kwargs_by_label = build_scorers(config, entailment_model)

    for label in config["methods_to_run"]:
        os.makedirs(os.path.join(raw_dir, label), exist_ok=True)

    recipe_concurrency = config.get("recipe_concurrency", 1)
    args_per_recipe = [
        (recipe_name, dags[recipe_name], data_dir, config, scorers, resolved_kwargs_by_label, cached_llm, raw_dir, baseline_raw_dir, dry_run)
        for recipe_name in recipe_names
    ]
    if recipe_concurrency > 1:
        _print_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=recipe_concurrency) as pool:
            futures = [pool.submit(process_recipe_invalid, *args) for args in args_per_recipe]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for args in args_per_recipe:
            process_recipe_invalid(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ares_topodev/configs/experiment_invalid.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(_abspath(args.config))
    run(config, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
