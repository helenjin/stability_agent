"""Does SAGER's tau converge as the topological-order budget L increases?

Motivation: across the 217-claim, 15-recipe merged dataset, 33 claims (15%)
have more than 3 true (depth-2) ancestors, where the number of distinct
relative orderings of just that ancestor subset can exceed the production
L=10 (k! grows fast: 4!=24, up to 10!=3,628,800 for the single largest case).
A direct check on the real graph (scrambledeggs node 16, 10 ancestors) found
L=10 gives 10/10 distinct restricted orderings with zero collisions, and even
L=1000 gives 999/1000 distinct -- i.e. L=10 is a small, essentially arbitrary
sample of a space nowhere near saturated. This script checks whether that
actually matters for tau, using the real backbone model, on the 9 real claims
with >=7 ancestors (the most extreme cases) across the 4 recipes that contain
them.

Design:
  - Same real model/adapter as the main experiment (Qwen2.5-7B-Instruct via
    ARESBackedEntailmentScorer), same depth (2), same seed (42) -- everything
    held fixed except L, which is swept over L_VALUES for each recipe.
  - N (num_soundness_samples) is reduced to 100 (from the production 200)
    purely to keep this diagnostic affordable -- held IDENTICAL across every
    L in the sweep, so it contributes an equal MC-noise floor at every point
    and does not confound the L-trend itself. MC convergence at this N was
    already separately validated on a toy graph in
    sager/validation/mc_convergence_validation.py; this script is checking L,
    not re-litigating N.
  - Only the 4 recipes containing a >=7-ancestor claim are run (not all 15/24)
    -- sager_known_graph computes every node in the recipe per call regardless,
    so this also captures every other high-ancestor node in the same recipe
    for free (e.g. scrambledeggs also has 2 more claims at 7 ancestors).

Output: prints tau + diagnostics for every (recipe, L) pair as it completes
(for live monitoring), then writes the full result table to
l_convergence_results.json next to this script.
"""
import importlib.util
import json
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from ares_topodev.eval_harness import _bootstrap  # noqa: F401


def _force_inert_vllm_stub():
    if "vllm" in sys.modules:
        return
    stubs_dir = os.path.join(REPO_ROOT, "ares_topodev", "vendor", "_stubs")
    spec = importlib.util.spec_from_file_location("vllm", os.path.join(stubs_dir, "vllm.py"))
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    sys.modules["vllm"] = stub


_force_inert_vllm_stub()

from ares_topodev.topo_reorder.dag import extract_recipe_dag, load_recipe_json
from ares_topodev.eval_harness.recipe_example import build_recipe_example
from sager import sager_known_graph
from sager.experiments.ares_vs_sager_topodev import (
    DATA_DIR, SEED, SAGER_DEPTH, TEMPERATURE, ARESBackedEntailmentScorer,
    build_entailment_model, build_graph_and_priors,
)

# --- config for this diagnostic only -----------------------------------
N_DIAGNOSTIC = 100  # reduced from production SAGER_N=200; see module docstring
L_VALUES = [10, 25, 50, 100]
# None -> QwenLLMFixed's own device_map="auto" (accelerate-managed). An
# explicit "cuda:0" currently raises inside this environment's installed
# transformers version ("model has been loaded with accelerate and therefore
# cannot be moved to a specific device") -- a pre-existing environment/library
# version issue unrelated to SAGER, not something to work around here since
# this script only runs one process at a time (no need for explicit pinning).
DEVICE = None

# recipe -> target node ids (>=7 true ancestors at depth=2), from the real
# 217-claim dataset's num_ancestors field
TARGETS = {
    "scrambledeggs": [16, 21, 22],       # 10, 7, 7 ancestors
    "broccolistirfry": [2, 11, 17],      # 8, 8, 7 ancestors
    "capresebruschetta": [4],            # 8 ancestors
    "cucumberraita": [6, 12],            # 8, 7 ancestors
}

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "l_convergence_results.json")


def main():
    print(f"Loading shared entailment model on {DEVICE} ...", flush=True)
    t0 = time.time()
    entailment_model, mode = build_entailment_model(device=DEVICE)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    results = []  # list of {recipe, node_id, num_ancestors, L, tau, diagnostics, wall_seconds}

    for recipe_name, node_ids in TARGETS.items():
        print(f"\n=== {recipe_name} (targets: {node_ids}) ===", flush=True)
        raw = load_recipe_json(os.path.join(DATA_DIR, f"{recipe_name}.json"))
        dag = extract_recipe_dag(raw, recipe_name)
        example = build_recipe_example(dag, raw, base_seed=SEED, raw_claims_shuffle_idx=0)
        node_ids_all = dag.derived_node_ids
        G, base_priors = build_graph_and_priors(dag)
        claims = {nid: example.derived_claims_by_node_id[nid] for nid in node_ids_all}
        shared_entailment_scorer = ARESBackedEntailmentScorer(
            entailment_model, mode, TEMPERATURE, raw_claims=example.raw_claims
        )

        for L in L_VALUES:
            torch.manual_seed(SEED)
            t0 = time.time()
            sager_result = sager_known_graph(
                G, claims=claims, base_priors=base_priors, entailment_scorer=shared_entailment_scorer,
                depth=SAGER_DEPTH, num_soundness_samples=N_DIAGNOSTIC, max_orderings=L, seed=SEED,
            )
            wall = time.time() - t0
            for nid in node_ids:
                row = {
                    "recipe": recipe_name,
                    "node_id": nid,
                    "L": L,
                    "N": N_DIAGNOSTIC,
                    "tau": sager_result.tau[nid],
                    "diagnostics": sager_result.diagnostics,
                    "wall_seconds": wall,
                }
                results.append(row)
                print(
                    f"  L={L:4d}  node={nid:3d}  tau={sager_result.tau[nid]:.4f}  "
                    f"unique_calls={sager_result.diagnostics['unique_entailment_calls']:5d}  "
                    f"cache_hits={sager_result.diagnostics['cache_hits']:6d}  "
                    f"wall={wall:6.1f}s",
                    flush=True,
                )

        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} rows -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
