import os

import pytest

from ares_topodev.analysis.detection_sensitivity import (
    compute_graph_sensitivity,
    precision_recall_f1,
    predicted_error_set,
    verify_ordering_is_valid_topo_sort,
)
from ares_topodev.analysis.threshold_cv import assign_folds, compute_cv_thresholds, threshold_for_recipe
from ares_topodev.analysis.topodev import load_raw_results
from ares_topodev.topo_reorder.dag import extract_recipe_dag, load_all_recipe_dags, load_recipe_json

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ares", "data", "recipe_graphs"
)
RESULTS_ARES_RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw", "ares"
)


def test_predicted_error_set_keys_by_node_id():
    scores = {"9": 0.05, "1": 0.9, "16": 0.02}
    predicted = predicted_error_set(scores, threshold=0.1)
    assert predicted == {"9", "16"}


def test_precision_recall_f1_normal_case():
    predicted = {"1", "2", "3"}
    ground_truth = {"2", "3", "4"}
    precision, recall, f1, p_zero, r_zero = precision_recall_f1(predicted, ground_truth)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert not p_zero and not r_zero


def test_precision_recall_f1_zero_predicted_positives():
    """Documented convention: empty predicted set -> precision=0.0 (not NaN,
    not skipped), recall well-defined (0 / nonzero ground truth = 0), f1=0."""
    predicted = set()
    ground_truth = {"1", "2"}
    precision, recall, f1, p_zero, r_zero = precision_recall_f1(predicted, ground_truth)
    assert precision == 0.0
    assert p_zero is True
    assert recall == 0.0
    assert r_zero is False
    assert f1 == 0.0


def test_precision_recall_f1_empty_ground_truth_defensive_branch():
    """Not observed in real CaptainCookRecipes data (checked: 0/24 recipes have
    an empty ground truth set), but implemented defensively: vacuously correct
    (recall=1.0) if predictions are also empty, else recall=0.0."""
    precision, recall, f1, p_zero, r_zero = precision_recall_f1(set(), set())
    assert r_zero is True
    assert recall == 1.0

    precision, recall, f1, p_zero, r_zero = precision_recall_f1({"1"}, set())
    assert r_zero is True
    assert recall == 0.0


def test_no_real_recipe_has_empty_ground_truth():
    """Encodes the manual check performed before implementing: every one of
    the 24 real CaptainCookRecipes examples has at least one ground-truth
    error node, so the empty-ground-truth branch above is never actually
    exercised by real data (defensive code, not a live path)."""
    if not os.path.isdir(RESULTS_ARES_RAW):
        pytest.skip("Experiment 1 raw results not present in this checkout")
    results = load_raw_results(os.path.join(os.path.dirname(RESULTS_ARES_RAW)), "ares")
    assert len(results) > 0
    for r in results:
        n_pos = sum(r["ground_truth_error_by_node_id"].values())
        assert n_pos > 0, f"{r['recipe_name']} has an empty ground-truth error set"


def test_cv_thresholds_never_use_a_fold_recipes_own_data():
    """The core anti-leakage guarantee: for every fold, the threshold used by
    recipes in that fold must have been computed only from OTHER folds'
    recipes. Verified directly against real Experiment 1 data."""
    if not os.path.isdir(RESULTS_ARES_RAW):
        pytest.skip("Experiment 1 raw results not present in this checkout")
    results = load_raw_results(os.path.join(os.path.dirname(RESULTS_ARES_RAW)), "ares")
    cv = compute_cv_thresholds(results, k=5, seed=42)

    from ares_topodev.analysis.error_recovery import choose_threshold

    for fold in range(cv.k):
        train_recipes = {r["recipe_name"] for r in results if cv.fold_assignment[r["recipe_name"]] != fold}
        test_recipes = {r["recipe_name"] for r in results if cv.fold_assignment[r["recipe_name"]] == fold}
        assert train_recipes.isdisjoint(test_recipes)
        # Recompute independently and confirm it matches what compute_cv_thresholds stored
        train_results = [r for r in results if r["recipe_name"] in train_recipes]
        expected_threshold = choose_threshold(train_results)
        assert cv.threshold_by_fold[fold] == expected_threshold


def test_fold_assignment_is_deterministic_and_covers_all_recipes():
    names = [f"recipe_{i}" for i in range(24)]
    a = assign_folds(names, k=5, seed=42)
    b = assign_folds(names, k=5, seed=42)
    assert a == b
    assert set(a.keys()) == set(names)
    assert set(a.values()) == {0, 1, 2, 3, 4}


def test_claim_alignment_by_id_not_position_in_flip_rate():
    """Construct a synthetic result where the SAME node id sits at different
    positions across two orderings but keeps the same score -- flip rate for
    that node must be 0. Then flip a different node's score across the
    threshold while also changing its position -- it must register as a flip
    regardless of where it moved to. This directly tests that alignment is by
    node_id, not by sequence position."""
    result = {
        "dataset": "captaincookrecipes",
        "recipe_name": "fake",
        "method": "ares",
        "model": "gpt-4o-mini",
        "seed": 42,
        "step_text_by_node_id": {"1": "claim A", "2": "claim B", "3": "claim C"},
        "ground_truth_error_by_node_id": {"1": 1, "2": 0, "3": 0},
        "orderings": [
            {
                "ordering_index": 0,
                "topo_order_step_ids": [1, 2, 3],
                "scores_by_node_id": {"1": 0.05, "2": 0.9, "3": 0.9},
            },
            {
                "ordering_index": 1,
                # node 1 moved from position 0 to position 2; score unchanged -> no flip
                "topo_order_step_ids": [2, 3, 1],
                "scores_by_node_id": {"1": 0.05, "2": 0.9, "3": 0.9},
            },
            {
                "ordering_index": 2,
                # node 3 moved AND its score crossed the threshold -> should flip
                "topo_order_step_ids": [3, 1, 2],
                "scores_by_node_id": {"1": 0.05, "2": 0.9, "3": 0.02},
            },
        ],
    }

    class _FakeCv:
        k = 1
        fold_assignment = {"fake": 0}
        threshold_by_fold = {0: 0.1}

    from ares_topodev.analysis.detection_sensitivity import compute_graph_sensitivity

    # Bypass real DAG validation (synthetic data) by monkeypatching the checker to a no-op.
    import ares_topodev.analysis.detection_sensitivity as ds

    original = ds.verify_ordering_is_valid_topo_sort
    ds.verify_ordering_is_valid_topo_sort = lambda dag, order: None
    try:
        summary, ordering_results = compute_graph_sensitivity(result, dag=None, cv=_FakeCv())
    finally:
        ds.verify_ordering_is_valid_topo_sort = original

    # node "1": score identical across all 3 orderings (always predicted positive) -> no flip
    # node "3": score crosses threshold in ordering 2 -> flips
    # node "2": never crosses threshold -> no flip
    # flip_rate should be 1/3 (only node "3" flips)
    assert summary.flip_rate == pytest.approx(1 / 3)


def test_verify_ordering_is_valid_topo_sort_rejects_a_real_violation():
    raw = load_recipe_json(os.path.join(DATA_DIR, "ramen.json"))
    dag = extract_recipe_dag(raw, "ramen")
    # Deliberately invalid: reverse the stored topo_order (start id excluded) --
    # guaranteed to violate at least one edge for any non-trivial DAG.
    from ares_topodev.topo_reorder.topo_sample import sample_orderings

    valid_order = sample_orderings(dag, k=1, seed=1).orderings[0]
    broken_order = list(reversed(valid_order))
    with pytest.raises(AssertionError):
        verify_ordering_is_valid_topo_sort(dag, broken_order)
    # and the real valid order must pass without raising
    verify_ordering_is_valid_topo_sort(dag, valid_order)
