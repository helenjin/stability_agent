import os

from ares_topodev.topo_reorder.dag import load_all_recipe_dags
from ares_topodev.topo_reorder.invalid_topo import (
    find_adjacent_edges,
    generate_invalid_orderings,
    swap_adjacent,
    violated_edges,
)
from ares_topodev.topo_reorder.topo_sample import assert_valid_topo_order, sample_orderings

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vendor",
    "ares",
    "data",
    "recipe_graphs",
)


def _all_dags():
    return load_all_recipe_dags(DATA_DIR)


def test_swap_violates_exactly_one_edge_across_all_recipes():
    """Mathematical invariant: swapping two adjacent elements changes the
    relative order of exactly that pair and nothing else, so exactly one
    edge can become violated. Verified empirically here, not just argued."""
    dags = _all_dags()
    total_cases = 0
    for name, dag in dags.items():
        base_ordering = sample_orderings(dag, k=1, seed=42).orderings[0]
        cases = generate_invalid_orderings(dag, base_ordering)
        assert len(cases) > 0, f"{name}: expected at least one adjacent edge to swap"
        for case in cases:
            assert case.all_violated_edges == [case.edge_violated], (
                f"{name}: swap for {case.edge_violated} violated {case.all_violated_edges}, "
                "expected exactly one violation"
            )
            total_cases += 1
    assert total_cases > 0


def test_swapped_ordering_is_a_permutation_of_the_original():
    dags = _all_dags()
    dag = dags["ramen"]
    base_ordering = sample_orderings(dag, k=1, seed=42).orderings[0]
    cases = generate_invalid_orderings(dag, base_ordering)
    for case in cases:
        assert sorted(case.ordering) == sorted(base_ordering)
        assert case.ordering != base_ordering  # something actually changed


def test_valid_ordering_has_no_violations():
    dags = _all_dags()
    dag = dags["ramen"]
    base_ordering = sample_orderings(dag, k=1, seed=42).orderings[0]
    assert_valid_topo_order(dag, base_ordering)  # sanity: it really is valid
    assert violated_edges(dag, base_ordering) == []


def test_find_adjacent_edges_matches_manual_check():
    dags = _all_dags()
    dag = dags["ramen"]
    base_ordering = sample_orderings(dag, k=1, seed=42).orderings[0]
    position = {nid: i for i, nid in enumerate(base_ordering)}
    adjacent = find_adjacent_edges(dag, base_ordering)
    for (u, v) in adjacent:
        assert (u, v) in dag.edges
        assert position[v] == position[u] + 1


def test_swap_adjacent_rejects_non_adjacent_pair():
    dags = _all_dags()
    dag = dags["ramen"]
    base_ordering = sample_orderings(dag, k=1, seed=42).orderings[0]
    # pick two nodes far apart in the ordering
    u, v = base_ordering[0], base_ordering[-1]
    try:
        swap_adjacent(base_ordering, u, v)
        assert False, "expected ValueError for non-adjacent pair"
    except ValueError:
        pass
