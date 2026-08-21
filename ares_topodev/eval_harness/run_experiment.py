"""Orchestrates the valid-reordering experiment on CaptainCookRecipes.

For each recipe example x each sampled valid topological ordering x each
method: build a `data_entry` via the UNMODIFIED
`exp_helpers.datasets.base.BaseDataset.get_data_entry` and score it with the
UNMODIFIED `exp_helpers.methods.get_stability_scorer(...).get_stability_rate`.
All new logic here is either (a) orchestration/looping, (b) remapping
positions back to permanent step ids, or (c) I/O -- none of it touches ARES's
scoring math.

Usage:
    python -m ares_topodev.eval_harness.run_experiment \\
        --config ares_topodev/configs/experiment.yaml [--limit N] [--dry-run]
"""
import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time

import yaml

from ares_topodev.eval_harness import _bootstrap  # noqa: F401  (sys.path + OPENAI_API_KEY placeholder)
from ares_topodev.eval_harness.cache import CachingLLM, DiskPromptCache
from ares_topodev.eval_harness.entailment_prompts import build_recipe_entailment_mode
from ares_topodev.eval_harness.mock_llm import MockLLM
from ares_topodev.eval_harness.recipe_example import apply_ordering, build_recipe_example
from ares_topodev.topo_reorder.dag import extract_recipe_dag, load_all_recipe_dags, load_recipe_json
from ares_topodev.topo_reorder.topo_sample import count_or_estimate_linear_extensions, sample_orderings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abspath(relative_to_repo_root: str) -> str:
    if os.path.isabs(relative_to_repo_root):
        return relative_to_repo_root
    return os.path.join(REPO_ROOT, relative_to_repo_root)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_entailment_model(config: dict, dry_run: bool):
    from exp_helpers.exp_configs import MODEL_CONFIGS
    from exp_helpers.models import EntailmentModel, get_llm

    if dry_run:
        base_llm = MockLLM()
    else:
        model_config = MODEL_CONFIGS[config["backbone_model"]]
        base_llm = get_llm(**model_config)

    cache_path = _abspath(config["cache_path"])
    cache = DiskPromptCache(cache_path)
    cached_llm = CachingLLM(base_llm, cache)

    entailment_model = EntailmentModel(
        llm=cached_llm,
        max_new_tokens=config["max_new_tokens"],
        batch_size=config.get("entailment_batch_size", 8),  # vendored default is 8; pure concurrency, no effect on epsilon/delta/scoring math
    )
    return entailment_model, cached_llm


def build_scorers(config: dict, entailment_model):
    from exp_helpers.exp_configs import METHOD_CONFIGS
    from exp_helpers.methods import get_stability_scorer

    custom_entailment_mode = build_recipe_entailment_mode(entailment_model)

    overrides = config.get("method_kwarg_overrides", {}) or {}

    scorers = {}
    resolved_kwargs_by_label = {}  # JSON-serializable record of what was actually used, per method
    for label in config["methods_to_run"]:
        config_key = config["method_configs"][label]
        method_config = METHOD_CONFIGS[config_key]
        kwargs = dict(method_config["kwargs"])
        kwargs.update(overrides.get(label, {}))  # e.g. relax epsilon/delta for a faster smoke test
        resolved_kwargs_by_label[label] = {k: v for k, v in kwargs.items() if k != "entailment_mode"}
        kwargs["entailment_mode"] = custom_entailment_mode
        kwargs["temperature"] = config.get("temperature", 0.0)
        scorers[label] = get_stability_scorer(method_config["method"], entailment_model, config["p"], **kwargs)
    return scorers, resolved_kwargs_by_label


def build_data_entry(raw_claims, derived_claims):
    """Calls the vendored, unmodified BaseDataset.get_data_entry as an unbound
    method -- it never reads `self`, so any placeholder object works. Verified
    against the source: the method body only uses its two list arguments."""
    from exp_helpers.datasets.base import BaseDataset

    dummy_self = BaseDataset.__new__(BaseDataset)
    return BaseDataset.get_data_entry(dummy_self, raw_claims, derived_claims)


_print_lock = None  # set to a threading.Lock() in run() when recipe_concurrency > 1


def _log(message: str):
    if _print_lock is not None:
        with _print_lock:
            print(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr)


def process_recipe(recipe_name, dag, data_dir, config, scorers, resolved_kwargs_by_label, cached_llm, raw_dir, dry_run):
    """Runs one recipe through every sampled ordering x every configured
    method, saving incrementally after each ordering. Independent across
    recipes (own dag/example/topo_result, writes only to its own output
    files), so this is safe to call concurrently for different recipe_names
    from a thread pool -- see `run()`'s recipe_concurrency option. This is a
    pure orchestration/parallelism change: it does not alter epsilon/delta,
    sampling, or scoring math at all.
    """
    t0 = time.time()
    raw = load_recipe_json(os.path.join(data_dir, f"{recipe_name}.json"))
    example = build_recipe_example(
        dag, raw, base_seed=config["seed"], raw_claims_shuffle_idx=config["raw_claims_shuffle_idx"]
    )

    topo_result = sample_orderings(dag, k=config["K"], seed=config["seed"], max_attempts=config["max_topo_attempts"])
    linear_ext = count_or_estimate_linear_extensions(dag, seed=config["seed"])

    # method -> ordering_index -> {node_id: score}
    per_method_orderings = {label: [] for label in config["methods_to_run"]}
    # method -> [{"ordering_index": i, "error": "..."}] -- a single malformed
    # LLM response (e.g. a method's expected output format not being followed)
    # must not abort the whole run for every recipe/method. Failures are
    # recorded and that (method, ordering) pair is skipped -- the method's
    # own num_orderings_used/orderings list simply ends up shorter, rather
    # than crashing process_recipe (and, via the thread pool, every other
    # in-flight recipe too).
    per_method_failures = {label: [] for label in config["methods_to_run"]}

    def write_results(is_complete: bool):
        """Writes the current (possibly partial) per-method JSON files.
        Called after every single ordering, not just once at the end, so a
        kill/crash mid-recipe never loses more than one ordering's worth
        of already-paid-for API calls -- `orderings` just reflects however
        many are done so far, and `is_complete`/`num_orderings_used` make
        partial files unambiguous rather than silently indistinguishable
        from a finished run with a smaller K.
        """
        for label in config["methods_to_run"]:
            out = {
                "dataset": "captaincookrecipes",
                "recipe_name": recipe_name,
                "method": label,
                "method_config_key": config["method_configs"][label],
                "model": "mock-llm" if dry_run else config["backbone_model"],
                "hyperparams": {
                    "p": config["p"],
                    "temperature": config.get("temperature", 0.0),
                    **resolved_kwargs_by_label[label],
                },
                "seed": config["seed"],
                "num_orderings_requested": topo_result.num_requested,
                "num_orderings_used": len(per_method_orderings[label]),
                "is_complete": is_complete,
                "topo_sampling_exhausted": topo_result.exhausted,
                "num_valid_orderings_estimate": {
                    "value": linear_ext.value,
                    "exact": linear_ext.exact,
                    "lower_bound_unique_found": linear_ext.lower_bound_unique_found,
                    "importance_sample_estimate": linear_ext.importance_sample_estimate,
                },
                "step_text_by_node_id": {str(k): v for k, v in example.derived_claims_by_node_id.items()},
                "ground_truth_error_by_node_id": {
                    str(k): v for k, v in example.ground_truth_error_by_node_id.items()
                },
                "orderings": per_method_orderings[label],
                "failed_orderings": per_method_failures[label],
            }
            out_path = os.path.join(raw_dir, label, f"{recipe_name}.json")
            tmp_path = out_path + f".tmp.{threading.get_ident()}"
            with open(tmp_path, "w") as f:
                json.dump(out, f, indent=2)
            os.replace(tmp_path, out_path)  # atomic within this filesystem -- never a half-written file on disk

    for ordering_index, order in enumerate(topo_result.orderings):
        derived_claims = apply_ordering(example, order)
        data_entry = build_data_entry(example.raw_claims, derived_claims)

        for label, scorer in scorers.items():
            try:
                result = scorer.get_stability_rate(data_entry)
                scores_by_node_id = {str(node_id): score for node_id, score in zip(order, result.stability_rates)}
            except Exception as e:  # noqa: BLE001 -- deliberately broad: a single malformed
                # LLM response from any vendored scorer must not abort every other
                # in-flight recipe/method. Logged and recorded, never silently dropped.
                _log(f"ERROR [{recipe_name}/{label}] ordering {ordering_index}: {type(e).__name__}: {e}")
                per_method_failures[label].append(
                    {"ordering_index": ordering_index, "error": f"{type(e).__name__}: {e}"}
                )
                continue
            per_method_orderings[label].append(
                {"ordering_index": ordering_index, "topo_order_step_ids": order, "scores_by_node_id": scores_by_node_id}
            )

        write_results(is_complete=(ordering_index == len(topo_result.orderings) - 1))
        _log(
            f"[{recipe_name}] ordering {ordering_index + 1}/{len(topo_result.orderings)} saved "
            f"(cache_hits={cached_llm.hits} cache_misses={cached_llm.misses})"
        )

    dt = time.time() - t0
    _log(
        f"[{recipe_name}] nodes={len(dag.derived_node_ids)} "
        f"orderings={len(topo_result.orderings)}/{topo_result.num_requested} "
        f"cache_hits={cached_llm.hits} cache_misses={cached_llm.misses} time={dt:.1f}s"
    )


def run(config: dict, limit=None, dry_run=False):
    global _print_lock

    data_dir = _abspath(config["recipe_data_dir"])
    results_dir = _abspath(config["results_dir"])
    raw_dir = os.path.join(results_dir, "raw")

    dags = load_all_recipe_dags(data_dir)
    recipe_names = sorted(dags.keys())
    if limit is not None:
        recipe_names = recipe_names[:limit]

    entailment_model, cached_llm = build_entailment_model(config, dry_run=dry_run)
    scorers, resolved_kwargs_by_label = build_scorers(config, entailment_model)

    for label in config["methods_to_run"]:
        os.makedirs(os.path.join(raw_dir, label), exist_ok=True)

    recipe_concurrency = config.get("recipe_concurrency", 1)
    if recipe_concurrency > 1:
        _print_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=recipe_concurrency) as pool:
            futures = {
                pool.submit(
                    process_recipe,
                    recipe_name,
                    dags[recipe_name],
                    data_dir,
                    config,
                    scorers,
                    resolved_kwargs_by_label,
                    cached_llm,
                    raw_dir,
                    dry_run,
                ): recipe_name
                for recipe_name in recipe_names
            }
            for future in concurrent.futures.as_completed(futures):
                recipe_name = futures[future]
                future.result()  # re-raises if process_recipe errored, instead of silently swallowing it
    else:
        for recipe_name in recipe_names:
            process_recipe(
                recipe_name,
                dags[recipe_name],
                data_dir,
                config,
                scorers,
                resolved_kwargs_by_label,
                cached_llm,
                raw_dir,
                dry_run,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ares_topodev/configs/experiment.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N recipes")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use a deterministic mock LLM instead of real API calls (no cost, structural validation only)",
    )
    args = parser.parse_args()

    config = load_config(_abspath(args.config))
    run(config, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
