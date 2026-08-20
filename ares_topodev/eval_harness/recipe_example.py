"""Build a topo-order-independent representation of one CaptainCookRecipes example.

Adapted from `RecipeGraphDatasetRaw3.__getitem__` in
`vendor/ares/src/exp_helpers/datasets/recipe_graph.py` (the class actually used
by the `recipe_graph3` data config, which is what
`ares_demo_captaincookrecipes.ipynb` runs). This is new code, not a
modification of the vendored file -- ares/src is untouched.

Two changes from the original `__getitem__`:
  1. It no longer hardcodes `data_example['topo_order']` as the presentation
     order. Instead every derived claim's text and ground-truth error label is
     keyed by its permanent step id, so any valid topo_order can be applied
     afterwards (see `apply_ordering` below) without recomputing anything.
  2. It drops the outer dataset-repetition/shuffle machinery
     (`RecipeGraphDatasetRaw3.__init__`'s `data * ceil(dataset_len/len(data))`
     followed by a dataset-level shuffle) since we process each recipe file
     once, not as an element of a repeated length-1000 dataset. `idx` in the
     original `sample_random_integer(..., seed=idx)` calls always used
     `self.seed` (constant, NOT `idx`) for the deleted-ingredient draw, and
     `idx+1` for the raw-claims shuffle -- we replicate both RNG calls exactly,
     fixing the raw-claims shuffle index to a constant (default 0) so a given
     recipe's raw_claims list (rules + remaining ingredient facts) is bit-for-bit
     identical across every topological reordering we evaluate. This is what
     satisfies the "same set of claims, same text, only derived-claim order
     changes" control -- raw_claims doesn't merely coincide across orderings,
     it is the *same computed value*, computed once per recipe and reused.

Per-node derived-claim text depends only on (node id, steps text, tgt2src),
none of which change with presentation order -- so `derived_claims_by_node_id`
is provably identical no matter which ordering is later applied
(see tests/test_recipe_example.py).
"""
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

from ares_topodev.topo_reorder.dag import RecipeDag


def _sample_random_integer(max_val: int, seed) -> int:
    local_rng = random.Random(seed)
    return local_rng.randint(0, max_val)


def _shuffle_list_with_local_rng(items: list, seed) -> list:
    local_rng = random.Random(seed)
    out = items[:]
    local_rng.shuffle(out)
    return out


@dataclass
class RecipeExample:
    recipe_name: str
    raw_claims: List[str]                       # base claims (rules + ingredient facts); fixed across all orderings
    derived_claims_by_node_id: Dict[int, str]    # text identical across ALL orderings, keyed by permanent step id
    ground_truth_error_by_node_id: Dict[int, int]
    deleted_ingredient: str
    deleted_ingredient_idx: int


def build_recipe_example(
    dag: RecipeDag,
    data_example_raw: dict,
    base_seed: int = 42,
    raw_claims_shuffle_idx: int = 0,
) -> RecipeExample:
    steps = {int(k): v for k, v in data_example_raw["steps"].items()}
    edges = [tuple(e) for e in data_example_raw["edges"]]
    ingredients = data_example_raw["ingredients"]
    all_mapped_ingredients_unique = list(data_example_raw["all_mapped_ingredients_unique"])
    if "None" in all_mapped_ingredients_unique:
        all_mapped_ingredients_unique.remove("None")
    step_mapped_ingredients = data_example_raw["step_mapped_ingredients"]

    sorted_step_ids = sorted(steps.keys())
    step_mapped_ingredients_dict = {
        key: (
            step_mapped_ingredients[i]
            if step_mapped_ingredients[i] is not None and step_mapped_ingredients[i][0] != "None"
            else None
        )
        for i, key in enumerate(sorted_step_ids)
    }
    ingredients_dict = {
        key: (ingredients[i] if ingredients[i] is not None and ingredients[i][0] != "None" else None)
        for i, key in enumerate(sorted_step_ids)
    }

    src2tgt: Dict[int, List[int]] = {nid: [] for nid in steps}
    tgt2src: Dict[int, List[int]] = {nid: [] for nid in steps}
    for (u, v) in edges:
        src2tgt[u].append(v)
        tgt2src[v].append(u)

    template = "Only after the necessary preceding steps ({}), And if we have all the ingredients, we can then {}."
    # NOTE: only nodes with >=1 predecessor get a rule, matching the original
    # `defaultdict(list)`-based `tgt2src.items()` iteration (nodes with no
    # incoming edge, e.g. START, never appear as keys there). Iteration order
    # here (by ascending step id) differs from the original (first-seen-while-
    # scanning-edges) order; this only affects which exact permutation
    # `remained_raw_claims` ends up in after the fixed-seed shuffle below, not
    # its content -- and since raw_claims is computed once per recipe and
    # reused unchanged across every ordering we evaluate, this doesn't affect
    # the experiment's controls.
    rules = [
        template.format(", and ".join([steps[p] for p in preds]), steps[tgt])
        for tgt, preds in tgt2src.items()
        if preds
    ]

    ingredient_states = [f"We have {item}." for item in all_mapped_ingredients_unique]
    deleted_ingredient_idx = _sample_random_integer(len(ingredient_states) - 1, base_seed)
    deleted_ingredient = ingredient_states[deleted_ingredient_idx]
    new_ingredient_states = (
        ingredient_states[:deleted_ingredient_idx] + ingredient_states[deleted_ingredient_idx + 1 :]
    )
    remained_raw_claims = rules + new_ingredient_states
    remained_raw_claims = _shuffle_list_with_local_rng(remained_raw_claims, raw_claims_shuffle_idx) + [
        "We now START."
    ]

    # Ground truth error propagation, keyed by permanent step id (not position).
    deleted_ingredient_name = all_mapped_ingredients_unique[deleted_ingredient_idx]
    initial_error_step_ids = [
        step_id
        for step_id in steps.keys()
        if ingredients_dict[step_id] is not None
        and deleted_ingredient_name in step_mapped_ingredients_dict[step_id]
    ]
    all_error_step_ids: List[int] = []
    dq = deque(initial_error_step_ids)
    while dq:
        step_id = dq.popleft()
        all_error_step_ids.append(step_id)
        for dstep in src2tgt.get(step_id, []):
            if dstep not in all_error_step_ids:
                dq.append(dstep)
    error_set = set(all_error_step_ids)

    derived_claims_by_node_id: Dict[int, str] = {}
    ground_truth_error_by_node_id: Dict[int, int] = {}
    for nid in dag.derived_node_ids:  # every node except START
        preds = tgt2src.get(nid, [])
        if preds:
            previous_steps_str = ", and ".join([steps[p] for p in preds])
        else:
            previous_steps_str = ""
        previous_steps_str = f"Because we have completed all previous steps ({previous_steps_str}),"
        if ingredients_dict.get(nid) is not None:
            necessary_ingredients_str = f'and have all necessary ingredients ({", and ".join(ingredients_dict[nid])}),'
        else:
            necessary_ingredients_str = ""
        derived_claims_by_node_id[nid] = (
            f"{previous_steps_str} {necessary_ingredients_str} we can now do the step {steps[nid]}. "
            f"And now we have completed this step {steps[nid]}."
        )
        ground_truth_error_by_node_id[nid] = 1 if nid in error_set else 0

    return RecipeExample(
        recipe_name=dag.recipe_name,
        raw_claims=remained_raw_claims,
        derived_claims_by_node_id=derived_claims_by_node_id,
        ground_truth_error_by_node_id=ground_truth_error_by_node_id,
        deleted_ingredient=deleted_ingredient,
        deleted_ingredient_idx=deleted_ingredient_idx,
    )


def apply_ordering(example: RecipeExample, topo_order: List[int]) -> List[str]:
    """Return the `derived_claims` list text in the given node-id order --
    same strings as any other ordering, just reindexed."""
    return [example.derived_claims_by_node_id[nid] for nid in topo_order]
