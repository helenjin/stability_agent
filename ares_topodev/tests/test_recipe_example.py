import os

import ares_topodev.eval_harness._bootstrap  # noqa: F401  (sets up sys.path for exp_helpers)
from exp_helpers.datasets.recipe_graph import RecipeGraphDatasetRaw3

from ares_topodev.eval_harness.recipe_example import apply_ordering, build_recipe_example
from ares_topodev.topo_reorder.dag import extract_recipe_dag, load_recipe_json
from ares_topodev.topo_reorder.topo_sample import sample_orderings

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vendor",
    "ares",
    "data",
    "recipe_graphs",
)


def _load(name):
    raw = load_recipe_json(os.path.join(DATA_DIR, f"{name}.json"))
    dag = extract_recipe_dag(raw, name)
    return raw, dag


def test_matches_vendored_ares_reference_implementation():
    """Our reimplementation, applied with the recipe's own stored topo_order,
    must reproduce the vendored RecipeGraphDatasetRaw3's derived_claims text
    and error labels exactly -- this is the fidelity check that justifies
    reusing our per-node text under arbitrary reorderings."""
    for name in ("ramen", "coffee", "zoodles"):
        raw, dag = _load(name)

        orig = RecipeGraphDatasetRaw3.__new__(RecipeGraphDatasetRaw3)
        orig.data = [raw]
        orig.dataset_len = 1
        orig.seed = 42
        item0 = RecipeGraphDatasetRaw3.__getitem__(orig, 0)

        ours = build_recipe_example(dag, raw, base_seed=42, raw_claims_shuffle_idx=1)
        ours_derived = apply_ordering(ours, raw["topo_order"][1:])

        assert item0["derived_claims"] == ours_derived, name
        assert item0["deleted_ingredient"] == ours.deleted_ingredient, name
        assert sorted(item0["remained_raw_claims"]) == sorted(ours.raw_claims), name
        expected_labels = [ours.ground_truth_error_by_node_id[nid] for nid in raw["topo_order"][1:]]
        assert item0["step_error_labels"] == expected_labels, name


def test_claim_text_is_identical_across_valid_reorderings():
    """The core control: reordering must never change any claim's text."""
    for name in ("ramen", "coffee"):
        raw, dag = _load(name)
        example = build_recipe_example(dag, raw, base_seed=42, raw_claims_shuffle_idx=0)
        orderings = sample_orderings(dag, k=5, seed=7).orderings
        assert len(orderings) >= 2

        text_by_node_across_orderings = {nid: set() for nid in dag.derived_node_ids}
        for order in orderings:
            claims_text = apply_ordering(example, order)
            for nid, text in zip(order, claims_text):
                text_by_node_across_orderings[nid].add(text)

        for nid, texts in text_by_node_across_orderings.items():
            assert len(texts) == 1, f"{name} node {nid}: claim text changed across orderings: {texts}"


def test_raw_claims_fixed_across_orderings_for_same_recipe():
    raw, dag = _load("ramen")
    ex1 = build_recipe_example(dag, raw, base_seed=42, raw_claims_shuffle_idx=0)
    ex2 = build_recipe_example(dag, raw, base_seed=42, raw_claims_shuffle_idx=0)
    assert ex1.raw_claims == ex2.raw_claims
    assert ex1.deleted_ingredient_idx == ex2.deleted_ingredient_idx


def test_orderings_permute_only_presentation_not_node_set():
    raw, dag = _load("ramen")
    example = build_recipe_example(dag, raw, base_seed=42)
    orderings = sample_orderings(dag, k=5, seed=1).orderings
    for order in orderings:
        assert sorted(order) == sorted(example.derived_claims_by_node_id.keys())
