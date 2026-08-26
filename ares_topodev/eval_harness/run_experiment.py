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
from collections import deque

import yaml

from ares_topodev.eval_harness import _bootstrap  # noqa: F401  (sys.path + OPENAI_API_KEY placeholder)
from ares_topodev.eval_harness.cache import CachingLLM, DiskPromptCache
from ares_topodev.eval_harness.entailment_prompts import build_recipe_entailment_mode
from ares_topodev.eval_harness.mock_llm import MockLLM
from ares_topodev.eval_harness.recipe_example import apply_ordering, build_recipe_example
from ares_topodev.eval_harness.sager import SagerStabilityScorer, build_graph_data_entry
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


def build_entailment_model(config: dict, dry_run: bool, device: str = None):
    from exp_helpers.exp_configs import MODEL_CONFIGS
    from exp_helpers.models import EntailmentModel, get_llm

    if dry_run:
        base_llm = MockLLM()
    else:
        model_config = MODEL_CONFIGS[config["backbone_model"]]
        if model_config.get("model_type") == "qwen":
            # The vendored QwenLLM crashes at temperature=0.0 (our standard
            # setting) and echoes the full prompt back in its output -- see
            # qwen_llm_fixed.py. Construct the fixed subclass directly
            # instead of the vendored loader's plain QwenLLM. `device`
            # (e.g. "cuda:1") pins this specific instance to one GPU, for
            # run()'s multi-GPU path below -- None means QwenLLMFixed's own
            # "auto" placement, unchanged from single-instance behavior.
            from ares_topodev.eval_harness.qwen_llm_fixed import QwenLLMFixed

            base_llm = QwenLLMFixed(
                device=device, **{k: v for k, v in model_config.items() if k != "model_type"}
            )
        else:
            base_llm = get_llm(**model_config)
        if model_config.get("model_type") == "openai":
            from ares_topodev.eval_harness import usage_tracker

            # Default scoped to THIS config's own results_dir, not a fixed
            # shared path -- experiment.yaml, experiment_sager.yaml, and
            # experiment_sager_ancestors.yaml all have different results_dir
            # values; defaulting to one fixed path would silently merge
            # every config's token usage into a single file with no way to
            # attribute cost back to which run produced it. Pass
            # usage_log_path explicitly if a shared log is actually wanted.
            default_usage_log_path = os.path.join(config["results_dir"], "usage_log.jsonl")
            usage_tracker.enable(_abspath(config.get("usage_log_path", default_usage_log_path)))

    cache_path = _abspath(config["cache_path"])
    cache = DiskPromptCache(cache_path)
    cached_llm = CachingLLM(base_llm, cache)

    entailment_model = EntailmentModel(
        llm=cached_llm,
        max_new_tokens=config["max_new_tokens"],
        batch_size=config.get("entailment_batch_size", 8),  # vendored default is 8; pure concurrency, no effect on epsilon/delta/scoring math
    )
    return entailment_model, cached_llm


# Methods whose vendored scorer class expects the raw LLM object (anything
# with a .generate() method) rather than the EntailmentModel wrapper --
# confirmed from methods/loader.py's own type hint (Union[EntailmentModel,
# BaseLLM]) and comment ("this can only work with LLM"). These scorers call
# llm.generate(...) directly with their own hand-written prompt; they don't
# do premise/hypothesis-pair entailment scoring at all, so our custom
# per-pair recipe entailment prompt and the entailment-model-specific
# `temperature` kwarg (LLMJudgeWholeStabilityScorer.__init__ has no such
# parameter) don't apply to them -- injecting either would be a category
# error, not a real config knob, and `temperature` would be an unexpected
# keyword argument at construction time.
METHODS_REQUIRING_RAW_LLM = {"llm_judge_whole"}

# SAGER (gold-graph) is not a vendored ARES method -- it lives in
# ares_topodev.eval_harness.sager, not vendor/ares, per the "create a new
# scorer rather than silently modifying ARES in place" rule. It is
# constructed directly rather than through the vendored
# `get_stability_scorer` loader, but resolves its epsilon/delta from the
# SAME `method_configs["ares"]` config-key kwargs `ares` itself uses (see
# `build_scorers` below), so the two methods are never accidentally run
# under different epsilon/delta -- required for the section-11 comparison to
# manipulate only the premise universe, nothing else.
METHODS_REQUIRING_GRAPH = {"sager", "sager_ancestors"}

# sager_ancestors is an exploratory variant: instead of Pa_G(c) (direct graph
# parents), it uses Anc_G(c) -- the full transitive-ancestor closure -- as
# the premise universe. Reuses SagerStabilityScorer/graph_tree_stability_rate
# completely unmodified: that code only ever asks "what does this dict say
# c's premise-defining nodes are," and has no notion of "direct" vs
# "transitive" baked in. Anc_G(c) is still a function of node identity only
# (never of a traversal order), so it's just as order-invariant as Pa_G(c) --
# this tests whether SAGER's F1 gap vs ARES is really about SPARSITY (too few
# premises) rather than about being graph-conditioned per se, while keeping
# everything else (canonical ordering, projection, broadcast, epsilon/delta)
# identical.


def compute_ancestors_by_node_id(dag):
    """Anc_G(c) = every node with a directed path to c, i.e. the transitive
    closure of Pa_G under dag.full_tgt2src -- excluding START (not a derived
    claim, same exclusion Pa_G(c) already applies) and excluding c itself."""
    ancestors_by_node_id = {}
    for nid in dag.derived_node_ids:
        visited = set()
        frontier = deque(p for p in dag.full_tgt2src.get(nid, []) if p != dag.start_id)
        while frontier:
            p = frontier.popleft()
            if p in visited:
                continue
            visited.add(p)
            frontier.extend(q for q in dag.full_tgt2src.get(p, []) if q != dag.start_id)
        ancestors_by_node_id[nid] = sorted(visited)
    return ancestors_by_node_id


def build_scorers(config: dict, entailment_model, cached_llm):
    from exp_helpers.exp_configs import METHOD_CONFIGS
    from exp_helpers.methods import get_stability_scorer

    custom_entailment_mode = build_recipe_entailment_mode(entailment_model)

    overrides = config.get("method_kwarg_overrides", {}) or {}

    scorers = {}
    resolved_kwargs_by_label = {}  # JSON-serializable record of what was actually used, per method
    for label in config["methods_to_run"]:
        if label in METHODS_REQUIRING_GRAPH:
            if "ares" not in config["method_configs"]:
                raise ValueError(
                    f"'{label}' resolves its epsilon/delta from method_configs['ares'] "
                    "(so it's guaranteed to share them with ares) -- 'ares' must "
                    "also be listed in method_configs, even if not in methods_to_run."
                )
            ares_config_key = config["method_configs"]["ares"]
            kwargs = {
                k: v for k, v in METHOD_CONFIGS[ares_config_key]["kwargs"].items() if k != "entailment_mode"
            }
            kwargs.update(overrides.get(label, {}))
            resolved_kwargs_by_label[label] = dict(kwargs)
            kwargs["entailment_mode"] = custom_entailment_mode
            kwargs["temperature"] = config.get("temperature", 0.0)
            kwargs["seed"] = config["seed"]
            scorers[label] = SagerStabilityScorer(entailment_model, p=config["p"], **kwargs)
            continue

        config_key = config["method_configs"][label]
        method_config = METHOD_CONFIGS[config_key]
        kwargs = dict(method_config["kwargs"])
        kwargs.update(overrides.get(label, {}))  # e.g. relax epsilon/delta for a faster smoke test
        resolved_kwargs_by_label[label] = {k: v for k, v in kwargs.items() if k != "entailment_mode"}
        if method_config["method"] in METHODS_REQUIRING_RAW_LLM:
            # Use this method's own kwargs completely unmodified (e.g. its own
            # binary/granular `entailment_mode` string, which selects between
            # its own two hardcoded system prompts -- not our custom recipe
            # entailment prompt object, and not a per-pair scoring mode).
            scorers[label] = get_stability_scorer(method_config["method"], cached_llm, config["p"], **kwargs)
        else:
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
    # Gold dependency graph, restricted to derived-claim nodes: Pa_G(c) for
    # SAGER, direct edges only (dag.full_tgt2src is built straight from the
    # recipe JSON's `edges`, independent of any topo_order -- see
    # topo_reorder/dag.py). START is excluded as a parent since it's not a
    # derived claim (mirrors BaseDataset.get_data_entry never turning START
    # into a premise "claim" either) -- a node whose only real predecessor is
    # START has Pa_G(c) = {} and is scored from raw claims only, same as any
    # other graph root.
    parents_by_node_id = {
        nid: [p for p in dag.full_tgt2src.get(nid, []) if p != dag.start_id] for nid in dag.derived_node_ids
    }
    # Anc_G(c): full transitive-ancestor closure, for the sager_ancestors
    # variant (see METHODS_REQUIRING_GRAPH's comment above). Computed unconditionally
    # (cheap for recipe-sized graphs) so it's always available for the raw
    # output's reference fields, regardless of whether sager_ancestors is
    # actually in methods_to_run.
    ancestors_by_node_id = compute_ancestors_by_node_id(dag)
    graph_by_label = {"sager": parents_by_node_id, "sager_ancestors": ancestors_by_node_id}
    # graph_by_label's keys must exactly track METHODS_REQUIRING_GRAPH -- a
    # future graph-based method added to one but not the other would
    # otherwise surface as a bare KeyError deep in the per-ordering loop
    # below instead of failing immediately, clearly, here.
    assert set(graph_by_label.keys()) == METHODS_REQUIRING_GRAPH, (
        f"graph_by_label keys {set(graph_by_label.keys())} != METHODS_REQUIRING_GRAPH {METHODS_REQUIRING_GRAPH}"
    )

    topo_result = sample_orderings(dag, k=config["K"], seed=config["seed"], max_attempts=config["max_topo_attempts"])
    linear_ext = count_or_estimate_linear_extensions(dag, seed=config["seed"])

    # Skip a label entirely -- no recompute, no rewrite -- if it already has a
    # complete result on disk for exactly this many orderings. Without this,
    # re-running a config that mixes an already-complete method (e.g. ares,
    # sager -- fast, cache-hit-only) with a genuinely new one (e.g.
    # sager_ancestors) would still re-walk the complete method's own ordering
    # loop from empty per_method_orderings, and write_results (below) writes
    # EVERY label in methods_to_run after every ordering -- so the already-
    # complete method's file gets overwritten with THIS run's own in-progress
    # partial state until it separately re-reaches the same K, transiently
    # (and, if this run is killed first, permanently) regressing a complete
    # file back to incomplete for no reason. seed/K/max_topo_attempts are
    # deterministic, so a file already at this exact K is trusted rather than
    # reproduced. (limit=len(topo_result.orderings), not num_requested: a DAG
    # with fewer than K valid orderings could have topo_sampling_exhausted
    # with fewer than num_requested -- that's still "as complete as this
    # config can get", not partial.)
    labels_to_run = []
    for label in config["methods_to_run"]:
        existing_path = os.path.join(raw_dir, label, f"{recipe_name}.json")
        skip = False
        if os.path.exists(existing_path):
            try:
                with open(existing_path) as f:
                    existing = json.load(f)
                skip = existing.get("is_complete") and existing.get("num_orderings_used") == len(topo_result.orderings)
            except (json.JSONDecodeError, OSError):
                skip = False  # a corrupt/half-written file is not "already complete" -- redo it
        if skip:
            _log(f"[{recipe_name}/{label}] already complete at {len(topo_result.orderings)} orderings -- skipping")
        else:
            labels_to_run.append(label)

    if not labels_to_run:
        _log(f"[{recipe_name}] all methods already complete -- nothing to do")
        return

    # method -> ordering_index -> {node_id: score}
    per_method_orderings = {label: [] for label in labels_to_run}
    # method -> [{"ordering_index": i, "error": "..."}] -- a single malformed
    # LLM response (e.g. a method's expected output format not being followed)
    # must not abort the whole run for every recipe/method. Failures are
    # recorded and that (method, ordering) pair is skipped -- the method's
    # own num_orderings_used/orderings list simply ends up shorter, rather
    # than crashing process_recipe (and, via the thread pool, every other
    # in-flight recipe too).
    per_method_failures = {label: [] for label in labels_to_run}

    def write_results(is_complete: bool):
        """Writes the current (possibly partial) per-method JSON files.
        Called after every single ordering, not just once at the end, so a
        kill/crash mid-recipe never loses more than one ordering's worth
        of already-paid-for API calls -- `orderings` just reflects however
        many are done so far, and `is_complete`/`num_orderings_used` make
        partial files unambiguous rather than silently indistinguishable
        from a finished run with a smaller K. Only writes labels_to_run --
        an already-complete label (see above) is never touched.
        """
        for label in labels_to_run:
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
                # Gold dependency graph, fixed per recipe (not per ordering) --
                # Pa_G(c) for every derived node. Present for every method's
                # output file (not just sager's) so ARES's raw files also
                # carry the graph for reference/comparison, even though ARES
                # itself never reads it.
                "parents_by_node_id": {str(k): v for k, v in parents_by_node_id.items()},
                "ancestors_by_node_id": {str(k): v for k, v in ancestors_by_node_id.items()},
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
        # `order` is reused here purely as a traversal/computation sequence for
        # SAGER (it is a valid topological order of the same graph) -- unlike
        # `data_entry` above, `graph_data_entry`'s premises are determined by
        # `parents_by_node_id`, never by position in `order`. This is what
        # Sanity Check 1 (topological invariance) exercises: the same K
        # orderings already sampled for ARES are reused as SAGER's traversal
        # orders too.
        graph_data_entry_by_label = {
            graph_label: build_graph_data_entry(
                example.raw_claims, order, example.derived_claims_by_node_id, graph_by_label[graph_label]
            )
            for graph_label in METHODS_REQUIRING_GRAPH
            if graph_label in labels_to_run
        }

        for label in labels_to_run:
            scorer = scorers[label]
            try:
                entry = graph_data_entry_by_label[label] if label in METHODS_REQUIRING_GRAPH else data_entry
                result = scorer.get_stability_rate(entry)
                # Most scorers loop directly over `ent_inputs` and append exactly
                # one score per input, so len(stability_rates) == len(order) holds
                # by construction. llm_judge_whole is different: it asks the LLM
                # for one JSON list covering every derived claim in a single call,
                # and nothing guarantees the model returns the same count it was
                # given. zip() truncates silently on a length mismatch, which
                # would misattribute scores to the wrong node ids without any
                # error -- so this is checked explicitly rather than trusted.
                if len(result.stability_rates) != len(order):
                    raise ValueError(
                        f"stability_rates length {len(result.stability_rates)} != "
                        f"order length {len(order)}"
                    )
                scores_by_node_id = {str(node_id): score for node_id, score in zip(order, result.stability_rates)}
            except Exception as e:  # noqa: BLE001 -- deliberately broad: a single malformed
                # LLM response from any vendored scorer must not abort every other
                # in-flight recipe/method. Logged and recorded, never silently dropped.
                _log(f"ERROR [{recipe_name}/{label}] ordering {ordering_index}: {type(e).__name__}: {e}")
                per_method_failures[label].append(
                    {"ordering_index": ordering_index, "error": f"{type(e).__name__}: {e}"}
                )
                continue
            ordering_record = {
                "ordering_index": ordering_index,
                "topo_order_step_ids": order,
                "scores_by_node_id": scores_by_node_id,
            }
            # Opt-in (default off -- this is real data volume, and every
            # existing run/config predates this field): the per-node queried
            # weighted-sample population SagerStabilityScorer already computes
            # to reach `stability_rate`, otherwise discarded once aggregated.
            # Lets a later analysis attribute how much any single premise
            # (e.g. one specific ancestor, for sager_ancestors) moved a node's
            # score -- by correlating that premise's retention bit against
            # query_y across these rows -- at zero additional model calls,
            # since it's the same computation already paid for, not a fresh
            # leave-one-out ablation (see conversation: that needs k+1 full
            # node-scorings per node and doesn't scale to a full dataset).
            if config.get("save_particle_data", False) and label in METHODS_REQUIRING_GRAPH:
                ordering_record["particle_data_by_node_id"] = {
                    str(r["node_id"]): {
                        "parent_ids": r["parent_ids"],
                        "premises_text": r["premises_text"],
                        "query_samples": r["query_samples"],
                        "query_y": r["query_y"],
                        "query_counts": r["query_counts"],
                    }
                    for r in result.stab_rate_results
                }
            per_method_orderings[label].append(ordering_record)

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


def _run_multi_gpu_qwen(config: dict, recipe_names, dags, data_dir: str, raw_dir: str, num_gpus: int):
    """One dedicated (entailment_model, cached_llm, scorers) set per GPU --
    Qwen2.5-7B comfortably fits on a single 20GB GPU (see qwen_llm_fixed.py),
    so this machine's otherwise-idle additional GPUs can each run their own
    model instance in parallel rather than sitting unused while one GPU
    serializes every recipe. `process_recipe` itself is reused completely
    unchanged -- each GPU worker just calls it sequentially over its own
    slice of recipes, exactly like the single-instance recipe_concurrency
    path does over the whole list.

    Cache safety note: each GPU worker gets its OWN DiskPromptCache instance
    pointed at the SAME file. Two workers computing the exact same prompt
    concurrently could each append a (harmless, duplicate) cache line -- not
    a correctness risk here specifically, since recipes (and therefore their
    prompts) are partitioned disjointly across workers, so this can't
    actually happen in practice, only in principle.
    """
    global _print_lock
    _print_lock = threading.Lock()

    gpu_setups = []
    for gpu_id in range(num_gpus):
        entailment_model, cached_llm = build_entailment_model(config, dry_run=False, device=f"cuda:{gpu_id}")
        scorers, resolved_kwargs_by_label = build_scorers(config, entailment_model, cached_llm)
        gpu_setups.append((cached_llm, scorers, resolved_kwargs_by_label))

    def process_bucket(gpu_id, bucket):
        cached_llm, scorers, resolved_kwargs_by_label = gpu_setups[gpu_id]
        for recipe_name in bucket:
            process_recipe(
                recipe_name, dags[recipe_name], data_dir, config, scorers,
                resolved_kwargs_by_label, cached_llm, raw_dir, dry_run=False,
            )

    buckets = [recipe_names[i::num_gpus] for i in range(num_gpus)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = [pool.submit(process_bucket, gpu_id, bucket) for gpu_id, bucket in enumerate(buckets) if bucket]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # re-raises if a bucket errored, instead of silently swallowing it


def run(config: dict, limit=None, dry_run=False):
    global _print_lock

    data_dir = _abspath(config["recipe_data_dir"])
    results_dir = _abspath(config["results_dir"])
    raw_dir = os.path.join(results_dir, "raw")

    dags = load_all_recipe_dags(data_dir)
    recipe_names = sorted(dags.keys())
    if limit is not None:
        recipe_names = recipe_names[:limit]

    for label in config["methods_to_run"]:
        os.makedirs(os.path.join(raw_dir, label), exist_ok=True)

    qwen_num_gpus = config.get("qwen_num_gpus", 1)  # opt-in: default 1 leaves every non-Qwen config's behavior untouched
    if not dry_run and qwen_num_gpus > 1:
        from exp_helpers.exp_configs import MODEL_CONFIGS

        if MODEL_CONFIGS.get(config["backbone_model"], {}).get("model_type") == "qwen":
            _run_multi_gpu_qwen(config, recipe_names, dags, data_dir, raw_dir, qwen_num_gpus)
            return

    entailment_model, cached_llm = build_entailment_model(config, dry_run=dry_run)
    scorers, resolved_kwargs_by_label = build_scorers(config, entailment_model, cached_llm)

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
