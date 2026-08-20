from ares_topodev.analysis.error_recovery import choose_threshold, compute_error_recovery


def _fake_result(ground_truth, orderings_scores):
    return {
        "dataset": "captaincookrecipes",
        "method": "fake",
        "recipe_name": "fake_recipe",
        "ground_truth_error_by_node_id": ground_truth,
        "orderings": [{"scores_by_node_id": scores} for scores in orderings_scores],
    }


def test_choose_threshold_separates_perfectly_separable_data():
    result = _fake_result(
        ground_truth={"1": 1, "2": 0, "3": 0},
        orderings_scores=[
            {"1": 0.1, "2": 0.9, "3": 0.8},
            {"1": 0.2, "2": 0.85, "3": 0.75},
        ],
    )
    threshold = choose_threshold([result])
    # Anything strictly between 0.2 and 0.75 works; the sweep should find a
    # threshold that perfectly separates error (low score) from sound (high score).
    for scores in (r["scores_by_node_id"] for r in result["orderings"]):
        predicted_error = {nid for nid, s in scores.items() if s < threshold}
        assert predicted_error == {"1"}


def test_perfectly_stable_predictions_give_jaccard_one_and_exact_match():
    ground_truth = {"1": 1, "2": 0, "3": 0}
    result = _fake_result(
        ground_truth=ground_truth,
        orderings_scores=[
            {"1": 0.0, "2": 1.0, "3": 1.0},
            {"1": 0.0, "2": 1.0, "3": 1.0},  # identical predicted set every ordering
            {"1": 0.0, "2": 1.0, "3": 1.0},
        ],
    )
    er = compute_error_recovery(result, threshold=0.5)
    assert er.mean_pairwise_jaccard == 1.0
    assert er.frac_orderings_exact_match_gt == 1.0
    assert er.mean_recall == 1.0
    assert er.mean_precision == 1.0


def test_unstable_predictions_give_lower_jaccard_and_partial_recall():
    ground_truth = {"1": 1, "2": 1, "3": 0}
    result = _fake_result(
        ground_truth=ground_truth,
        orderings_scores=[
            {"1": 0.0, "2": 1.0, "3": 1.0},  # only flags node 1 -> misses node 2
            {"1": 0.0, "2": 0.0, "3": 1.0},  # flags both -> exact match with ground truth
        ],
    )
    er = compute_error_recovery(result, threshold=0.5)
    assert er.mean_pairwise_jaccard < 1.0  # predicted sets differ ({"1"} vs {"1","2"})
    assert 0.0 < er.mean_recall < 1.0
    assert er.frac_orderings_exact_match_gt == 0.5  # only the second ordering matches ground truth exactly


def test_no_ground_truth_errors_gives_perfect_recall_by_convention():
    result = _fake_result(
        ground_truth={"1": 0, "2": 0},
        orderings_scores=[{"1": 1.0, "2": 1.0}],
    )
    er = compute_error_recovery(result, threshold=0.5)
    assert er.mean_recall == 1.0  # vacuously true: no true errors to miss
    assert er.frac_orderings_exact_match_gt == 1.0  # predicted empty set == ground truth empty set
