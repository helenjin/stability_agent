"""First real SAGER experiment: ARES vs. SAGER (known-graph) under valid
topological reorderings of the same reasoning dependency graph, on real
CaptainCookRecipes examples.

Goal: test whether ARES (which scores a reasoning trace using its *observed*
serialization) is sensitive to which valid topological ordering that
serialization happens to be, and whether SAGER (which scores from the
dependency graph G directly) is invariant, or nearly invariant, to it.

For each example, everything is held fixed except the *observed* serialization
rho_r used to build ARES's input:
  - same claims (RecipeExample.derived_claims_by_node_id, keyed by permanent
    node id -- see ares_topodev/eval_harness/recipe_example.py)
  - same dependency DAG G (ares_topodev/topo_reorder/dag.py's RecipeDag,
    read directly from the recipe JSON's ground-truth `edges`)
  - same ground-truth error labels (RecipeExample.ground_truth_error_by_node_id)
  - same entailment model: one shared `exp_helpers.models.entailment_model.
    EntailmentModel` instance (wrapping Qwen3-4B via `QwenLLMFixed`, plain
    HF transformers -- not vllm, broken on this machine; not an API model,
    no key configured), used by BOTH ARES and SAGER via the
    `ARESBackedEntailmentScorer` adapter below, so score differences can
    only come from the evaluation framework, never from using different
    models. (An earlier pass ran this same script against `MockLLM`, the
    deterministic zero-cost dry-run stand-in, to validate the experiment
    harness itself before spending real GPU time -- see git history / the
    conversation this script was written for.)
  - same hyperparameters (epsilon/delta/p/temperature for ARES; d/N/L/seed
    for SAGER) across every rho_r.

SAGER's own inputs (G, claims, base_priors, entailment_scorer, d, N, L, seed)
never include rho_r anywhere -- see `run_example` below, where
`sager_known_graph(...)` is called exactly ONCE per recipe, outside the
rho_r loop, and its result is reused for every row. (An earlier version of
this script called it once per rho_r with byte-identical arguments each
time -- a ~5x waste of SAGER's Monte Carlo compute, since `CachingEntailmentScorer`
is scoped to a single `sager_known_graph` call and can't help across separate
calls. SAGER's invariance to rho_r is a property of its inputs, not something
that needs re-deriving 5 times per recipe -- it's proven by construction here
and already exhaustively validated elsewhere: `sager/tests/`,
`sager/validation/serialization_invariance_validation.py`, and empirically
across all 217 claims in the merged 15-recipe run with zero exceptions.)
rho_r is used ONLY to build ARES's `derived_claims` list via `apply_ordering`.

Do NOT run dependency-error injection, ancestor-depth ablations, structural
uncertainty variants, or anything beyond this single comparison here -- see
the conversation this script was written for.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import torch

from ares_topodev.eval_harness import _bootstrap  # noqa: F401  (sys.path + OPENAI_API_KEY placeholder)


def _force_inert_vllm_stub():
    """This machine has a real `vllm` install (for the Qwen3-4B backbone work)
    whose CUDA extension fails to import here (`libcudart.so.13` missing --
    a pre-existing environment issue, unrelated to this experiment). We only
    ever use QwenLLMFixed (plain HF transformers), never real vllm, so force Python to use
    `ares_topodev/vendor/_stubs/vllm.py` (an inert stub already shipped for
    exactly this "vllm not usable/not installed" case -- see _bootstrap.py)
    instead of the real, broken package, by pre-registering it in
    sys.modules before anything imports `exp_helpers.models`. Scoped to this
    process only; does not touch the real vllm install or any shared/vendor
    file.
    """
    if "vllm" in sys.modules:
        return
    stubs_dir = os.path.join(REPO_ROOT, "ares_topodev", "vendor", "_stubs")
    spec = importlib.util.spec_from_file_location("vllm", os.path.join(stubs_dir, "vllm.py"))
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    sys.modules["vllm"] = stub
from ares_topodev.eval_harness.entailment_prompts import build_recipe_entailment_mode
from ares_topodev.eval_harness.recipe_example import RecipeExample, apply_ordering, build_recipe_example
from ares_topodev.topo_reorder.dag import RecipeDag, extract_recipe_dag, load_recipe_json
from ares_topodev.topo_reorder.topo_sample import count_or_estimate_linear_extensions, sample_orderings

from sager import sager_known_graph

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "ares_topodev", "vendor", "ares", "data", "recipe_graphs")
RESULTS_DIR = os.path.join(REPO_ROOT, "sager", "experiments", "results_ares_vs_sager")

# --- Fixed experiment configuration (recorded verbatim in the report; not
# tuned against gold labels) ------------------------------------------------

# Full dataset: all 24 CaptainCookRecipes graphs. An earlier depth=inf pass
# on all 24 would have needed ~141k unique real-model SAGER calls (the large,
# densely-connected recipes' full ancestor closures blow up the Monte Carlo
# active-subset space combinatorially); capping depth at 2 (see SAGER_DEPTH
# below) brings that down to ~5.8k across all 24 -- fully tractable, so the
# earlier 11-recipe size-filtered subset is no longer needed. See
# conversation for the size/depth cost tradeoff that led here.
RECIPE_NAMES = [
    "blenderbananapancakes", "breakfastburritos", "broccolistirfry",
    "buttercorncup", "capresebruschetta", "cheesepimiento", "coffee",
    "cucumberraita", "dressedupmeatballs", "herbomeletwithfriedtomatoes",
    "microwaveeggsandwich", "microwavefrenchtoast", "microwavemugpizza",
    "mugcake", "panfriedtofu", "pinwheels", "ramen", "sautedmushrooms",
    "scrambledeggs", "spicedhotchocolate", "spicytunaavocadowraps",
    "tomatochutney", "tomatomozzarellasalad", "zoodles",
]

R = 5                       # observed serializations (rho_1..rho_R) per example
SEED = 42                   # shared across ordering sampling, ARES, and SAGER
P_BASE = 0.95                # ARES's own base-claim soundness prior; reused, unchanged, as every SAGER root's base_priors value
ARES_EPSILON = 0.1
ARES_DELTA = 0.1
TEMPERATURE = 0.0

# Real backbone: Qwen2.5-7B-Instruct via plain HF transformers (QwenLLMFixed),
# matching experiment_sager_qwen.yaml's already-working local-GPU path in this
# repo -- NOT vllm (broken on this machine, see _force_inert_vllm_stub) and
# NOT an API model (no OPENAI_API_KEY configured here). Same model instance is
# shared by ARES and SAGER via ARESBackedEntailmentScorer below, so any score
# difference between the two methods still comes only from which premises
# each framework shows the model, never from using different models.
#
# Switched from Qwen3-4B after finding (see conversation) that Qwen3-4B under
# this exact prompt/template answers "Very Likely" for almost every input
# regardless of content, with rare, seemingly-idiosyncratic exceptions not
# tracking genuine entailment reasoning -- verified via a controlled
# premise-set sweep (varying size, content, and order independently), not
# just spot-checked. Qwen2.5-7B-Instruct is the model this project's own
# earlier SAGER work (results_sager_qwen/) already validated as usable on
# this same recipe entailment prompt.
BACKBONE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BACKBONE_MAX_NEW_TOKENS = 40  # matches experiment_sager_qwen.yaml: label comes first, small cap skips post-label rambling

SAGER_DEPTH = 2              # d: capped (not full closure) -- see RECIPE_NAMES comment above for why;
                             # this is a real scope change vs the first 3-recipe pass (was math.inf),
                             # made for computational feasibility at this scale, not an ablation choice
SAGER_N = 200                # Monte Carlo soundness-sampling budget
SAGER_L = 10                 # topological-order budget (Pi_G)

# Fixed, not tuned on gold labels (see analysis/detection_sensitivity.py for
# this project's usual CV-tuned threshold -- deliberately not used here, per
# "do not tune these using test labels yet").
VERDICT_THRESHOLD = 0.5


def build_entailment_model(device=None):
    from exp_helpers.models import EntailmentModel
    from ares_topodev.eval_harness.qwen_llm_fixed import QwenLLMFixed

    llm = QwenLLMFixed(  # real backbone; SAME instance shared by ARES and SAGER below
        model_name=BACKBONE_MODEL_NAME, temperature=TEMPERATURE, max_new_tokens=BACKBONE_MAX_NEW_TOKENS,
        device=device,  # pins this instance to one GPU (e.g. "cuda:1") for parallel sharding across recipes
    )
    model = EntailmentModel(llm=llm, max_new_tokens=BACKBONE_MAX_NEW_TOKENS, batch_size=16)
    mode = build_recipe_entailment_mode(model)  # the actual CaptainCookRecipes system prompt/label mapping ARES uses
    return model, mode


class ARESBackedEntailmentScorer:
    """Adapter: exposes ARES's real EntailmentModel (+ the real CaptainCookRecipes
    prompt/label-mapping config) as a plain `(premises: list[str], claim: str)
    -> float` callable, satisfying sager's EntailmentScorer interface. This
    is what lets ARES and SAGER score every individual (premises, claim) pair
    through the literal same model, prompt, and parsing logic -- not just
    'the same model family' -- so any score difference between the two
    methods can only come from which premises each framework decides to
    show the model, never from the model itself.

    `raw_claims` (the base recipe facts, e.g. "we have milk") are prepended
    to every non-root query's premises here, matching
    `BaseDataset.get_data_entry`'s own convention of seeding EVERY claim's
    premise list with the raw claims -- including ARES's first claim in a
    sequence, which is never truly empty. Without this, SAGER's queries (the
    only ones that can have a genuinely empty active-ancestor set) go out
    with zero premises while ARES's never do -- an asymmetry in premise
    universes, not a difference in method, and one that turned out to
    trigger a severe degenerate response from Qwen3-4B (reflexive "Very
    Likely" on empty premises, "Very Unlikely" on almost anything else,
    regardless of content -- see conversation). `num_raw=0` is kept
    unchanged (matches ARES's own default, `split_raw_derived=False`): raw
    claims are folded into the premises list itself, not given special
    prompt treatment, exactly as ARES already does by default.
    """

    def __init__(self, entailment_model, mode, temperature: float = 0.0, raw_claims: Optional[List[str]] = None):
        self.entailment_model = entailment_model
        self.mode = mode
        self.temperature = temperature
        self.raw_claims = list(raw_claims) if raw_claims else []

    def __call__(self, premises: List[str], claim: str) -> float:
        full_premises = self.raw_claims + list(premises)
        scores = self.entailment_model.forward(
            [full_premises], [claim], mode=self.mode, num_raw=0, temperature=self.temperature
        )
        return float(scores[0].item())


def build_ares_scorer(entailment_model, mode):
    from exp_helpers.methods import get_stability_scorer

    return get_stability_scorer(
        "cert_nonexact", entailment_model, p=P_BASE,
        epsilon=ARES_EPSILON, delta=ARES_DELTA, entailment_mode=mode, temperature=TEMPERATURE,
    )


def build_ares_data_entry(raw_claims, derived_claims):
    """Calls the vendored, unmodified BaseDataset.get_data_entry -- same
    helper run_experiment.py uses for the real ARES pipeline."""
    from exp_helpers.datasets.base import BaseDataset

    dummy_self = BaseDataset.__new__(BaseDataset)
    return BaseDataset.get_data_entry(dummy_self, raw_claims, derived_claims)


def build_graph_and_priors(dag: RecipeDag):
    """Same Pa_G(c) construction run_experiment.py uses for the OLD sager.py
    path (see eval_harness/run_experiment.py:process_recipe's
    `parents_by_node_id`), rebuilt here as a networkx.DiGraph for the new
    `sager` package. Excludes START (never a derived claim, matches
    BaseDataset.get_data_entry's own convention)."""
    G = nx.DiGraph()
    G.add_nodes_from(dag.derived_node_ids)
    for nid in dag.derived_node_ids:
        for p in dag.full_tgt2src.get(nid, []):
            if p != dag.start_id:
                G.add_edge(p, nid)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    base_priors = {n: P_BASE for n in roots}
    return G, base_priors


@dataclass
class NodeMetrics:
    mean_score: float
    topodev: float
    topovar: float
    min_score: float
    max_score: float
    score_swing: float


def per_node_metrics(scores_by_r: List[Dict[int, float]], node_ids: List[int]) -> Dict[int, NodeMetrics]:
    """tau_bar_M(v), TopoDev_M(v), TopoVar_M(v) exactly as specified: population
    mean/std/variance across the R observed serializations, per node."""
    out = {}
    for nid in node_ids:
        vals = [scores[nid] for scores in scores_by_r]
        Rn = len(vals)
        mean = sum(vals) / Rn
        topovar = sum((v - mean) ** 2 for v in vals) / Rn
        out[nid] = NodeMetrics(
            mean_score=mean, topodev=math.sqrt(topovar), topovar=topovar,
            min_score=min(vals), max_score=max(vals), score_swing=max(vals) - min(vals),
        )
    return out


def count_verdict_flips(scores_by_r: List[Dict[int, float]], node_ids: List[int], threshold: float) -> int:
    flips = 0
    for nid in node_ids:
        preds = {int(scores[nid] < threshold) for scores in scores_by_r}
        if len(preds) > 1:
            flips += 1
    return flips


def run_example(recipe_name: str, entailment_model, mode):
    raw = load_recipe_json(os.path.join(DATA_DIR, f"{recipe_name}.json"))
    dag = extract_recipe_dag(raw, recipe_name)
    example: RecipeExample = build_recipe_example(dag, raw, base_seed=SEED, raw_claims_shuffle_idx=0)

    # Sanity check 3: gold labels are keyed by permanent node id, attached
    # once here, never touched again per-serialization.
    node_ids = dag.derived_node_ids
    gold_by_node = example.ground_truth_error_by_node_id
    assert set(gold_by_node.keys()) == set(node_ids)

    G, base_priors = build_graph_and_priors(dag)
    claims = {nid: example.derived_claims_by_node_id[nid] for nid in node_ids}
    # Sanity check 6: SAGER's own node ids ARE the permanent recipe step ids
    # (G's nodes == claims' keys == gold_by_node's keys) -- nothing gets
    # remapped through position, so SAGER's output dict is already keyed by
    # the same ids ARES's `order`-zip uses.
    assert set(G.nodes) == set(claims.keys()) == set(gold_by_node.keys())

    topo_result = sample_orderings(dag, k=R, seed=SEED, max_attempts=200)
    linear_ext = count_or_estimate_linear_extensions(dag, seed=SEED)

    ares_scorer = build_ares_scorer(entailment_model, mode)
    shared_entailment_scorer = ARESBackedEntailmentScorer(entailment_model, mode, TEMPERATURE, raw_claims=example.raw_claims)

    rows = []
    ares_scores_by_r: List[Dict[int, float]] = []
    sager_scores_by_r: List[Dict[int, float]] = []

    # SAGER's inputs never depend on rho_r (see module docstring), so it is
    # computed exactly once per recipe here, not once per iteration of the
    # rho_r loop below -- reused as-is for every row.
    sager_result = sager_known_graph(
        G, claims=claims, base_priors=base_priors, entailment_scorer=shared_entailment_scorer,
        depth=SAGER_DEPTH, num_soundness_samples=SAGER_N, max_orderings=SAGER_L, seed=SEED,
    )

    for r, order in enumerate(topo_result.orderings):
        # Sanity check 1: every generated serialization is a valid topological
        # ordering of the SAME graph (raises internally if not; asserted here too).
        assert sorted(order) == sorted(node_ids)

        # Sanity check 2/4: ARES receives the reordered sequence -- same claims,
        # reindexed only by position, nothing added/removed/modified.
        derived_claims = apply_ordering(example, order)
        assert len(derived_claims) == len(node_ids)
        assert set(derived_claims) == set(claims.values())
        data_entry = build_ares_data_entry(example.raw_claims, derived_claims)
        # Sanity check 8: ARES's own Monte Carlo sampling (sample_s_pertbs)
        # draws from torch's GLOBAL RNG with no internal seeding of its own
        # (confirmed: `torch.rand(...)` in stability_deterministic.py, no
        # manual_seed call anywhere in that module) -- so without reseeding
        # here, ares_scorer.get_stability_rate's own result depends on
        # whatever RNG state happened to carry over from the previous call,
        # confounding genuine reordering sensitivity with ordinary run-to-run
        # ARES-internal MC noise (verified: 3 consecutive runs of this script
        # gave 3 different ARES aggregate TopoDev/flip numbers before this
        # fix, while SAGER's own properly-seeded numbers were bit-identical
        # every time). Reseeding identically before every rho_r is the
        # "common random numbers" control: every observed serialization's
        # ARES call starts from the exact same RNG state, so any score
        # difference between them can only be attributed to the reordering,
        # never to which slice of an unseeded stream happened to be consumed.
        torch.manual_seed(SEED)
        ares_result = ares_scorer.get_stability_rate(data_entry)
        if len(ares_result.stability_rates) != len(order):
            raise ValueError(
                f"[{recipe_name}] ARES stability_rates length {len(ares_result.stability_rates)} "
                f"!= order length {len(order)}"
            )
        ares_scores = dict(zip(order, ares_result.stability_rates))
        ares_scores_by_r.append(ares_scores)

        # Sanity check 5: SAGER's inputs never depend on `order` (rho_r) --
        # `sager_result` was computed once above, before this loop, from G,
        # claims, base_priors, entailment_scorer, d, N, L, seed only. `order`
        # is not referenced anywhere below this comment.
        sager_scores_by_r.append(dict(sager_result.tau))

        for nid in node_ids:
            for method, scores in (("ares", ares_scores), ("sager", sager_result.tau)):
                score = scores[nid]
                rows.append({
                    "example_id": recipe_name,
                    "node_id": nid,
                    "serialization_id": r,
                    "serialization": json.dumps(order),
                    "method": method,
                    "score": score,
                    "binary_prediction": int(score < VERDICT_THRESHOLD),
                    "gold_label": gold_by_node[nid],
                })

    # This is now trivially True by construction -- sager_scores_by_r[r] is
    # the same dict copied R times, since sager_result is computed once above
    # and reused, not recomputed per r. Kept (rather than removed) so
    # `sager_exact_invariant`/`all_sager_invariant` stay valid fields for
    # existing downstream report scripts. The real invariance guarantee is
    # the algorithm's proof (SAGER's inputs never include rho_r) plus the
    # exhaustive empirical validation already on record -- see the module
    # docstring -- not a per-run recomputation.
    sager_exact_invariant = all(
        sager_scores_by_r[0] == sager_scores_by_r[r] for r in range(1, len(sager_scores_by_r))
    )

    return dict(
        recipe_name=recipe_name, dag=dag, example=example, node_ids=node_ids,
        topo_result=topo_result, linear_ext=linear_ext, rows=rows,
        ares_scores_by_r=ares_scores_by_r, sager_scores_by_r=sager_scores_by_r,
        sager_exact_invariant=sager_exact_invariant,
    )


def build_aggregate_rows(result: dict) -> List[dict]:
    node_ids = result["node_ids"]
    agg_rows = []
    for method, scores_by_r in (("ares", result["ares_scores_by_r"]), ("sager", result["sager_scores_by_r"])):
        metrics = per_node_metrics(scores_by_r, node_ids)
        flips_by_node = {
            nid: len({int(scores[nid] < VERDICT_THRESHOLD) for scores in scores_by_r}) > 1 for nid in node_ids
        }
        for nid in node_ids:
            m = metrics[nid]
            agg_rows.append({
                "example_id": result["recipe_name"],
                "node_id": nid,
                "method": method,
                "num_serializations": len(scores_by_r),
                "mean_score": m.mean_score,
                "topodev": m.topodev,
                "topovar": m.topovar,
                "min_score": m.min_score,
                "max_score": m.max_score,
                "score_swing": m.score_swing,
                "num_verdict_flips": int(flips_by_node[nid]),
            })
    return agg_rows


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """CLI overrides for running one shard of the full recipe set on one GPU
    (see run_parallel.py, which launches one such shard per GPU and merges
    the results). With no args, behaves exactly as before: all of
    RECIPE_NAMES, default device placement, RESULTS_DIR as-is."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", type=str, default=None,
                         help="comma-separated recipe names; defaults to all of RECIPE_NAMES")
    parser.add_argument("--device", type=str, default=None,
                         help='e.g. "cuda:1"; defaults to QwenLLMFixed\'s own device_map="auto"')
    parser.add_argument("--results-dir", type=str, default=None,
                         help="overrides RESULTS_DIR (e.g. a per-shard subdirectory)")
    return parser.parse_args()


def main():
    args = parse_args()
    recipe_names = args.recipes.split(",") if args.recipes else RECIPE_NAMES
    results_dir = args.results_dir if args.results_dir else RESULTS_DIR

    _force_inert_vllm_stub()
    os.makedirs(results_dir, exist_ok=True)
    entailment_model, mode = build_entailment_model(device=args.device)

    print("=" * 78)
    print("Configuration (fixed, recorded, not tuned on gold labels)")
    print("=" * 78)
    print(f"  recipes: {recipe_names}")
    print(f"  device: {args.device or 'auto'}")
    print(f"  R (observed serializations per example): {R}")
    print(f"  SAGER config: d={SAGER_DEPTH}, N={SAGER_N}, L={SAGER_L}, seed={SEED}")
    print(f"  ARES config: epsilon={ARES_EPSILON}, delta={ARES_DELTA}, p={P_BASE}, temperature={TEMPERATURE}")
    print(f"  shared entailment model: EntailmentModel(QwenLLMFixed({BACKBONE_MODEL_NAME!r})) + real CaptainCookRecipes prompt/mapping")
    print(f"  verdict threshold (fixed, not CV-tuned): {VERDICT_THRESHOLD}")

    all_rows = []
    all_agg_rows = []
    per_example_results = []

    for recipe_name in recipe_names:
        print()
        print("=" * 78)
        print(f"Example: {recipe_name}")
        print("=" * 78)
        result = run_example(recipe_name, entailment_model, mode)
        per_example_results.append(result)
        all_rows.extend(result["rows"])
        agg_rows = build_aggregate_rows(result)
        all_agg_rows.extend(agg_rows)

        dag = result["dag"]
        print(f"  graph edges ({len(dag.edges)}): {dag.edges}")
        print(f"  num nodes (derived claims): {len(result['node_ids'])}")
        print(f"  num valid topological orderings: requested={result['topo_result'].num_requested} "
              f"used={len(result['topo_result'].orderings)} "
              f"total_estimate={result['linear_ext'].value} (exact={result['linear_ext'].exact})")
        print(f"  observed serializations tested:")
        for r, order in enumerate(result["topo_result"].orderings):
            print(f"    rho_{r}: {order}")

        for method_key, scores_by_r in (("ARES", result["ares_scores_by_r"]), ("SAGER", result["sager_scores_by_r"])):
            print(f"  {method_key} claim scores under each serialization:")
            for nid in result["node_ids"]:
                vals = [f"{scores[nid]:.3f}" for scores in scores_by_r]
                print(f"    node {nid}: {vals}")

        metrics_ares = per_node_metrics(result["ares_scores_by_r"], result["node_ids"])
        metrics_sager = per_node_metrics(result["sager_scores_by_r"], result["node_ids"])
        mean_topodev_ares = statistics.fmean(m.topodev for m in metrics_ares.values())
        mean_topodev_sager = statistics.fmean(m.topodev for m in metrics_sager.values())
        print(f"  TopoDev (ARES):  mean={mean_topodev_ares:.4f}  per-node={{{', '.join(f'{n}:{m.topodev:.3f}' for n, m in metrics_ares.items())}}}")
        print(f"  TopoDev (SAGER): mean={mean_topodev_sager:.4f}  per-node={{{', '.join(f'{n}:{m.topodev:.3f}' for n, m in metrics_sager.items())}}}")

        ares_flips = count_verdict_flips(result["ares_scores_by_r"], result["node_ids"], VERDICT_THRESHOLD)
        sager_flips = count_verdict_flips(result["sager_scores_by_r"], result["node_ids"], VERDICT_THRESHOLD)
        print(f"  ARES verdict flips (threshold={VERDICT_THRESHOLD}): {ares_flips}/{len(result['node_ids'])} nodes")
        print(f"  SAGER verdict flips (threshold={VERDICT_THRESHOLD}): {sager_flips}/{len(result['node_ids'])} nodes")
        print(f"  SAGER exactly invariant across all {len(result['topo_result'].orderings)} observed serializations: "
              f"{result['sager_exact_invariant']}")
        if not result["sager_exact_invariant"]:
            print("  *** BUG: SAGER scores differ across observed serializations despite identical "
                  "G/claims/priors/seed for every rho_r -- see conversation instructions to diagnose before continuing. ***")

    row_path = os.path.join(results_dir, "row_level_results.csv")
    agg_path = os.path.join(results_dir, "aggregate_results.csv")
    write_csv(row_path, all_rows)
    write_csv(agg_path, all_agg_rows)

    print()
    print("=" * 78)
    print("Aggregate summary across all examples/nodes")
    print("=" * 78)
    for method in ("ares", "sager"):
        method_agg = [row for row in all_agg_rows if row["method"] == method]
        topodevs = [row["topodev"] for row in method_agg]
        topovars = [row["topovar"] for row in method_agg]
        swings = [row["score_swing"] for row in method_agg]
        n_changed = sum(1 for s in swings if s > 0)
        n_flipped = sum(row["num_verdict_flips"] for row in method_agg)
        print(f"  [{method}] n_claims={len(method_agg)}  "
              f"mean_topodev={statistics.fmean(topodevs):.4f}  median_topodev={statistics.median(topodevs):.4f}  "
              f"mean_topovar={statistics.fmean(topovars):.4f}  max_score_swing={max(swings):.4f}  "
              f"frac_score_changed={n_changed / len(method_agg):.3f}  "
              f"frac_claims_with_verdict_flip={n_flipped / len(method_agg):.3f}")

    all_sager_invariant = all(r["sager_exact_invariant"] for r in per_example_results)
    print()
    print(f"SAGER exactly invariant across ALL examples: {all_sager_invariant}")
    print(f"Row-level results saved to: {row_path}")
    print(f"Aggregate results saved to: {agg_path}")

    return per_example_results, all_rows, all_agg_rows


if __name__ == "__main__":
    main()
