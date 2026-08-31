"""Tests for ares_topodev.sager_known_graph -- SAGER, known dependency graph
setting. Six required categories: linear chain, branching DAG, diamond DAG
(depth-limited ancestors), topological-serialization invariance, the L cap,
and reproducibility. Uses MockEntailmentScorer throughout (deterministic, no
external calls).
"""
import math

import networkx as nx
import pytest

from ares_topodev.sager_known_graph import (
    MockEntailmentScorer,
    compute_depth_limited_ancestors,
    restrict_order,
    sager_known_graph,
    sample_distinct_topological_orders,
)


# --- 1. Linear chain: a -> b -> c ---------------------------------------------


def test_linear_chain_exactly_one_ordering():
    G = nx.DiGraph([("a", "b"), ("b", "c")])
    orders = sample_distinct_topological_orders(G, max_orders=10, seed=0)
    assert orders == [["a", "b", "c"]]


def test_linear_chain_depth_limited_ancestors():
    G = nx.DiGraph([("a", "b"), ("b", "c")])
    assert compute_depth_limited_ancestors(G, "c", depth=1) == {"b"}  # parents only
    assert compute_depth_limited_ancestors(G, "c", depth=math.inf) == {"a", "b"}  # all ancestors
    assert compute_depth_limited_ancestors(G, "b", depth=1) == {"a"}
    assert compute_depth_limited_ancestors(G, "a", depth=math.inf) == set()


def test_linear_chain_sager_runs_unambiguously():
    G = nx.DiGraph([("a", "b"), ("b", "c")])
    claims = {"a": "flour and water are available", "b": "the dough is mixed", "c": "the bread is baked"}
    result = sager_known_graph(
        G,
        claims=claims,
        base_priors={"a": 0.9},
        entailment_scorer=MockEntailmentScorer(),
        depth=math.inf,
        num_soundness_samples=50,
        max_orderings=10,
        seed=1,
    )
    assert set(result.tau) == {"a", "b", "c"}
    assert result.diagnostics["num_topological_orders_used"] == 1
    for v in result.tau.values():
        assert 0.0 <= v <= 1.0


# --- 2. Branching DAG: a -> c, b -> c ------------------------------------------


def test_branching_dag_valid_orders():
    G = nx.DiGraph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edges_from([("a", "c"), ("b", "c")])
    orders = sample_distinct_topological_orders(G, max_orders=10, seed=0)
    order_set = {tuple(o) for o in orders}
    assert order_set == {("a", "b", "c"), ("b", "a", "c")}


def test_branching_dag_restrict_order():
    order = ["a", "b", "c"]
    assert restrict_order(order, {"a", "c"}) == ["a", "c"]
    assert restrict_order(order, {"b"}) == ["b"]
    assert restrict_order(order, set()) == []
    assert restrict_order(order, {"a", "b", "c"}) == ["a", "b", "c"]

    order2 = ["b", "a", "c"]
    assert restrict_order(order2, {"a", "c"}) == ["a", "c"]  # relative order from order2, not insertion order


# --- 3. Diamond DAG: a->b, a->c, b->d, c->d ------------------------------------


def _diamond():
    G = nx.DiGraph()
    G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    return G


def test_diamond_depth_limited_ancestors():
    G = _diamond()
    assert compute_depth_limited_ancestors(G, "d", depth=1) == {"b", "c"}
    assert compute_depth_limited_ancestors(G, "d", depth=2) == {"a", "b", "c"}
    assert compute_depth_limited_ancestors(G, "d", depth=math.inf) == {"a", "b", "c"}
    assert compute_depth_limited_ancestors(G, "b", depth=1) == {"a"}
    assert compute_depth_limited_ancestors(G, "a", depth=1) == set()


# --- 4. Topological-serialization invariance -----------------------------------


def test_topological_serialization_invariance():
    claims = {
        "a": "we have flour and water",
        "b": "the dough is mixed",
        "c": "the dough has risen",
        "d": "the bread is baked",
    }
    base_priors = {"a": 0.9}
    kwargs = dict(
        claims=claims,
        base_priors=base_priors,
        entailment_scorer=MockEntailmentScorer(),
        depth=math.inf,
        num_soundness_samples=100,
        max_orderings=10,
        seed=7,
    )

    G1 = nx.DiGraph()
    G1.add_nodes_from(["a", "b", "c", "d"])
    G1.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])

    G2 = nx.DiGraph()
    G2.add_nodes_from(["d", "c", "b", "a"])  # reversed node-insertion order
    G2.add_edges_from([("c", "d"), ("b", "d"), ("a", "c"), ("a", "b")])  # reordered edge insertion

    assert set(G1.edges) == set(G2.edges)  # same abstract DAG, different serialization

    result1 = sager_known_graph(G1, **kwargs)
    result2 = sager_known_graph(G2, **kwargs)

    assert result1.tau.keys() == result2.tau.keys()
    for node in result1.tau:
        assert math.isclose(result1.tau[node], result2.tau[node], rel_tol=1e-9, abs_tol=1e-12), (
            f"node {node}: {result1.tau[node]} != {result2.tau[node]}"
        )


# --- 5. L cap --------------------------------------------------------------------


def test_l_cap_all_returned_when_total_orderings_below_cap():
    G = nx.DiGraph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edges_from([("a", "c"), ("b", "c")])  # exactly 2 valid orderings
    orders = sample_distinct_topological_orders(G, max_orders=5, seed=0)
    assert len(orders) == 2
    assert len({tuple(o) for o in orders}) == 2  # no duplicates


def test_l_cap_enforced_and_deduplicated_for_large_ordering_space():
    # 4 independent roots feeding one sink -> 4! = 24 valid orderings, well above L.
    G = nx.DiGraph()
    roots = ["r1", "r2", "r3", "r4"]
    for r in roots:
        G.add_edge(r, "sink")
    L = 6
    orders = sample_distinct_topological_orders(G, max_orders=L, seed=123)
    assert len(orders) <= L
    assert len({tuple(o) for o in orders}) == len(orders)  # no duplicates
    for order in orders:
        assert order[-1] == "sink"
        assert set(order[:-1]) == set(roots)


# --- 6. Reproducibility -----------------------------------------------------------


def test_reproducibility_same_seed_identical_results():
    G = _diamond()
    claims = {"a": "flour and water", "b": "dough mixed", "c": "dough risen", "d": "bread baked"}
    kwargs = dict(
        claims=claims,
        base_priors={"a": 0.85},
        entailment_scorer=MockEntailmentScorer(),
        depth=math.inf,
        num_soundness_samples=80,
        max_orderings=8,
        seed=99,
    )
    result_a = sager_known_graph(G, **kwargs)
    result_b = sager_known_graph(G, **kwargs)
    assert result_a.tau == result_b.tau
    assert result_a.diagnostics == result_b.diagnostics


def test_reproducibility_different_seeds_may_differ_but_stay_in_unit_interval():
    G = _diamond()
    claims = {"a": "flour and water", "b": "dough mixed", "c": "dough risen", "d": "bread baked"}
    base_kwargs = dict(
        claims=claims,
        base_priors={"a": 0.85},
        entailment_scorer=MockEntailmentScorer(),
        depth=math.inf,
        num_soundness_samples=80,
        max_orderings=8,
    )
    result_1 = sager_known_graph(G, seed=1, **base_kwargs)
    result_2 = sager_known_graph(G, seed=2, **base_kwargs)

    for result in (result_1, result_2):
        for v in result.tau.values():
            assert 0.0 <= v <= 1.0


# --- Additional coverage: caching diagnostics, debug mode, validation ------------


def test_caching_diagnostics_reports_hits():
    G = _diamond()
    claims = {"a": "flour and water", "b": "dough mixed", "c": "dough risen", "d": "bread baked"}
    result = sager_known_graph(
        G,
        claims=claims,
        base_priors={"a": 0.9},
        entailment_scorer=MockEntailmentScorer(),
        depth=math.inf,
        num_soundness_samples=200,
        max_orderings=4,
        seed=3,
    )
    diag = result.diagnostics
    assert diag["entailment_requests"] >= diag["unique_entailment_calls"]
    assert diag["cache_hits"] == diag["entailment_requests"] - diag["unique_entailment_calls"]
    assert diag["cache_hits"] > 0  # repeated MC samples should recur


def test_debug_mode_reports_per_sample_p_alpha_active_set():
    G = nx.DiGraph([("a", "b")])
    claims = {"a": "premise", "b": "conclusion"}
    N = 10
    result = sager_known_graph(
        G,
        claims=claims,
        base_priors={"a": 1.0},
        entailment_scorer=MockEntailmentScorer(),
        depth=1,
        num_soundness_samples=N,
        max_orderings=5,
        seed=5,
        debug=True,
    )
    assert result.debug_samples is not None
    assert len(result.debug_samples) == N
    for sample in result.debug_samples:
        assert set(sample) == {"a", "b"}
        assert sample["a"]["A"] is None  # base node has no active-ancestor set
        assert sample["a"]["alpha"] in (0, 1)
        assert sample["b"]["A"] is None or set(sample["b"]["A"]) <= {"a"}


def test_debug_mode_off_by_default():
    G = nx.DiGraph([("a", "b")])
    claims = {"a": "premise", "b": "conclusion"}
    result = sager_known_graph(
        G,
        claims=claims,
        base_priors={"a": 1.0},
        entailment_scorer=MockEntailmentScorer(),
        num_soundness_samples=5,
        max_orderings=3,
        seed=5,
    )
    assert result.debug_samples is None


def test_missing_base_prior_raises():
    G = nx.DiGraph([("a", "b")])
    claims = {"a": "premise", "b": "conclusion"}
    with pytest.raises(ValueError):
        sager_known_graph(
            G, claims=claims, base_priors={}, entailment_scorer=MockEntailmentScorer(), seed=0
        )


def test_cyclic_graph_raises():
    G = nx.DiGraph([("a", "b"), ("b", "a")])
    with pytest.raises(ValueError):
        sager_known_graph(
            G,
            claims={"a": "x", "b": "y"},
            base_priors={},
            entailment_scorer=MockEntailmentScorer(),
            seed=0,
        )


def test_out_of_range_score_is_clipped():
    def bad_scorer(premises, claim):
        return 5.0  # out of [0, 1]

    G = nx.DiGraph([("a", "b")])
    result = sager_known_graph(
        G,
        claims={"a": "x", "b": "y"},
        base_priors={"a": 0.5},
        entailment_scorer=bad_scorer,
        num_soundness_samples=5,
        max_orderings=2,
        seed=0,
    )
    assert result.tau["b"] == 1.0  # clipped from 5.0
