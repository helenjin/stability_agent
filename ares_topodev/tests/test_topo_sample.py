import os

from ares_topodev.topo_reorder.dag import load_all_recipe_dags
from ares_topodev.topo_reorder.topo_sample import (
    assert_valid_topo_order,
    count_or_estimate_linear_extensions,
    sample_orderings,
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vendor",
    "ares",
    "data",
    "recipe_graphs",
)


def _all_dags():
    return load_all_recipe_dags(DATA_DIR)


def test_all_recipes_load_with_unambiguous_start_end():
    dags = _all_dags()
    assert len(dags) == 24
    for name, dag in dags.items():
        assert dag.steps[dag.start_id] == "START", name
        assert dag.steps[dag.end_id] == "END", name
        assert dag.end_id in dag.derived_node_ids, name
        assert dag.start_id not in dag.derived_node_ids, name


def test_sampled_orderings_are_all_valid_and_deduped():
    dags = _all_dags()
    for name, dag in dags.items():
        result = sample_orderings(dag, k=10, seed=42)
        assert len(result.orderings) >= 1, name
        assert len(result.orderings) <= 10
        seen = set()
        for order in result.orderings:
            assert_valid_topo_order(dag, order)  # raises on violation
            key = tuple(order)
            assert key not in seen, f"{name}: duplicate ordering returned"
            seen.add(key)
            assert order[-1] == dag.end_id, f"{name}: END not pinned last"
            assert sorted(order) == sorted(dag.derived_node_ids)


def test_invalid_ordering_is_rejected():
    dags = _all_dags()
    dag = dags["ramen"]
    order = sample_orderings(dag, k=1, seed=42).orderings[0]
    # Swap two adjacent nodes that violate a real dependency edge, if possible.
    position = {nid: i for i, nid in enumerate(order)}
    violated = False
    for (u, v) in dag.edges:
        if u == dag.start_id or v == dag.end_id:
            continue
        if position[u] < position[v]:
            broken = list(order)
            iu, iv = broken.index(u), broken.index(v)
            broken[iu], broken[iv] = broken[iv], broken[iu]
            try:
                assert_valid_topo_order(dag, broken)
            except AssertionError:
                violated = True
                break
    assert violated, "expected at least one edge-violating swap to be rejected"


def test_seed_is_reproducible():
    dags = _all_dags()
    dag = dags["ramen"]
    r1 = sample_orderings(dag, k=10, seed=123)
    r2 = sample_orderings(dag, k=10, seed=123)
    assert r1.orderings == r2.orderings


def test_different_seeds_can_differ():
    dags = _all_dags()
    dag = dags["ramen"]
    r1 = sample_orderings(dag, k=10, seed=1)
    r2 = sample_orderings(dag, k=10, seed=2)
    assert r1.orderings != r2.orderings


def test_linear_extension_count_is_at_least_num_sampled():
    dags = _all_dags()
    for name, dag in dags.items():
        result = sample_orderings(dag, k=10, seed=42)
        count = count_or_estimate_linear_extensions(dag, seed=42)
        assert count.value >= len(result.orderings) - 1e-6, name
