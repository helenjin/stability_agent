"""Experiment 2: does error-DETECTION performance (not just raw score) depend
on which valid topological ordering presents the same reasoning graph?

Reuses Experiment 1's already-computed, already-verified valid orderings and
scores directly (`results*/raw/<method>/<recipe>.json`) -- no new model
inference for ARES/Entail-Prev/Entail-Base. Thresholds come from
`analysis/threshold_cv.py`'s recipe-level 5-fold CV, never from the same
data being scored.

    Precision_M^pi = |Y_hat ∩ Y_G| / |Y_hat|     (0 if |Y_hat| == 0, see below)
    Recall_M^pi    = |Y_hat ∩ Y_G| / |Y_G|
    F1_M^pi        = 2PR / (P+R)                  (0 if P+R == 0)

    DeltaF1_M(G) = max_pi F1_M^pi - min_pi F1_M^pi     (over the sampled Topo_K(G))

Edge-case convention (documented, not silently chosen): if a method predicts
zero positive nodes for some ordering, precision is mathematically undefined
(0/0). We set precision=0.0 in that case (the standard scikit-learn
`zero_division=0` convention) and count how many times it happens, rather
than skip those cases or silently default some other way. Ground-truth-empty
graphs (Y_G = {}) would make recall undefined; we checked all 24
CaptainCookRecipes examples and none have an empty Y_G (0 occurrences), so
that branch is implemented defensively but never actually exercised here --
recall=1.0 if Y_hat is also empty (vacuously correct), else 0.0.
"""
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ares_topodev.analysis.threshold_cv import CvThresholds, threshold_for_recipe
from ares_topodev.topo_reorder.dag import RecipeDag
from ares_topodev.topo_reorder.topo_sample import assert_valid_topo_order


def predicted_error_set(scores_by_node_id: Dict[str, float], threshold: float) -> set:
    return {nid for nid, score in scores_by_node_id.items() if score < threshold}


def precision_recall_f1(predicted: set, ground_truth: set) -> Tuple[float, float, float, bool, bool]:
    """Returns (precision, recall, f1, precision_was_zero_division, recall_was_zero_division)."""
    tp = len(predicted & ground_truth)

    precision_zero_div = len(predicted) == 0
    precision = (tp / len(predicted)) if not precision_zero_div else 0.0

    recall_zero_div = len(ground_truth) == 0
    if recall_zero_div:
        recall = 1.0 if len(predicted) == 0 else 0.0
    else:
        recall = tp / len(ground_truth)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, precision_zero_div, recall_zero_div


@dataclass
class OrderingDetectionResult:
    ordering_index: int
    topo_order_step_ids: List[int]
    predicted_error_set: List[str]
    precision: float
    recall: float
    f1: float
    precision_zero_division: bool
    recall_zero_division: bool


def verify_ordering_is_valid_topo_sort(dag: RecipeDag, topo_order_step_ids: List[int]) -> None:
    """Explicit, independent re-verification that every tested ordering respects
    every dependency edge -- (u,v) in E => position(u) < position(v). Reuses
    Experiment 1's own checker rather than re-deriving it, but is called fresh
    here rather than assumed, per the "verify programmatically" requirement.
    Raises AssertionError on violation (never silently continues)."""
    assert_valid_topo_order(dag, topo_order_step_ids)


def compute_ordering_detection_results(
    result: dict, dag: RecipeDag, threshold: float
) -> List[OrderingDetectionResult]:
    ground_truth = {nid for nid, label in result["ground_truth_error_by_node_id"].items() if label == 1}
    out = []
    for ordering in result["orderings"]:
        verify_ordering_is_valid_topo_sort(dag, ordering["topo_order_step_ids"])
        predicted = predicted_error_set(ordering["scores_by_node_id"], threshold)
        precision, recall, f1, p_zero, r_zero = precision_recall_f1(predicted, ground_truth)
        out.append(
            OrderingDetectionResult(
                ordering_index=ordering["ordering_index"],
                topo_order_step_ids=ordering["topo_order_step_ids"],
                predicted_error_set=sorted(predicted),
                precision=precision,
                recall=recall,
                f1=f1,
                precision_zero_division=p_zero,
                recall_zero_division=r_zero,
            )
        )
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


@dataclass
class GraphOrderingSensitivity:
    method: str
    recipe_name: str
    fold: int
    threshold: float
    num_orderings: int
    mean_f1: float
    min_f1: float
    max_f1: float
    delta_f1: float
    mean_pairwise_jaccard: float
    exact_match_rate: float
    flip_rate: float
    num_zero_predicted_positive_orderings: int
    ground_truth_error_set: List[str] = field(default_factory=list)


def compute_graph_sensitivity(
    result: dict,
    dag: RecipeDag,
    cv: CvThresholds,
    ordering_results: Optional[List[OrderingDetectionResult]] = None,
) -> Tuple[GraphOrderingSensitivity, List[OrderingDetectionResult]]:
    recipe_name = result["recipe_name"]
    threshold = threshold_for_recipe(cv, recipe_name)
    fold = cv.fold_assignment[recipe_name]

    if ordering_results is None:
        ordering_results = compute_ordering_detection_results(result, dag, threshold)

    f1s = [o.f1 for o in ordering_results]
    predicted_sets = [set(o.predicted_error_set) for o in ordering_results]
    ground_truth = {nid for nid, label in result["ground_truth_error_by_node_id"].items() if label == 1}

    pairwise = []
    for i in range(len(predicted_sets)):
        for j in range(i + 1, len(predicted_sets)):
            pairwise.append(_jaccard(predicted_sets[i], predicted_sets[j]))
    mean_pairwise_jaccard = statistics.fmean(pairwise) if pairwise else 1.0

    exact_matches = sum(1 for p in predicted_sets if p == ground_truth)

    # Flip rate: for each claim (by permanent node id), does its predicted
    # sound/unsound label ever differ across the sampled orderings? Aligned
    # by node id, never by position -- predicted_sets are already keyed by
    # node id internally (sets of node-id strings), so this is automatic.
    all_node_ids = set(result["step_text_by_node_id"].keys())
    flips = 0
    for nid in all_node_ids:
        labels = {nid in p for p in predicted_sets}
        if len(labels) > 1:
            flips += 1
    flip_rate = flips / len(all_node_ids) if all_node_ids else 0.0

    num_zero_pred_pos = sum(1 for o in ordering_results if len(o.predicted_error_set) == 0)

    summary = GraphOrderingSensitivity(
        method=result["method"],
        recipe_name=recipe_name,
        fold=fold,
        threshold=threshold,
        num_orderings=len(ordering_results),
        mean_f1=statistics.fmean(f1s),
        min_f1=min(f1s),
        max_f1=max(f1s),
        delta_f1=max(f1s) - min(f1s),
        mean_pairwise_jaccard=mean_pairwise_jaccard,
        exact_match_rate=exact_matches / len(predicted_sets) if predicted_sets else 0.0,
        flip_rate=flip_rate,
        num_zero_predicted_positive_orderings=num_zero_pred_pos,
        ground_truth_error_set=sorted(ground_truth),
    )
    return summary, ordering_results


def bootstrap_ci_over_graphs(values: List[float], num_resamples: int = 10000, seed: int = 42, alpha: float = 0.05):
    """Graph-level (not ordering-level) bootstrap: resamples entire graphs
    with replacement, since the K orderings of one graph are repeated
    measurements, not independent draws. Never resamples orderings."""
    import random

    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(num_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * num_resamples)
    hi_idx = int((1 - alpha / 2) * num_resamples) - 1
    return (means[lo_idx], means[hi_idx])


@dataclass
class CrossGraphSummary:
    method: str
    n_graphs: int
    mean_f1_mean: float
    mean_f1_ci: Tuple[float, float]
    delta_f1_mean: float
    delta_f1_median: float
    delta_f1_ci: Tuple[float, float]
    mean_jaccard: float
    mean_exact_match_rate: float
    mean_flip_rate: float
    total_zero_predicted_positive_orderings: int


def summarize_across_graphs(graph_summaries: List[GraphOrderingSensitivity]) -> CrossGraphSummary:
    if not graph_summaries:
        raise ValueError("summarize_across_graphs called with no graphs")
    mean_f1_values = [g.mean_f1 for g in graph_summaries]
    delta_f1_values = [g.delta_f1 for g in graph_summaries]
    return CrossGraphSummary(
        method=graph_summaries[0].method,
        n_graphs=len(graph_summaries),
        mean_f1_mean=statistics.fmean(mean_f1_values),
        mean_f1_ci=bootstrap_ci_over_graphs(mean_f1_values, seed=42),
        delta_f1_mean=statistics.fmean(delta_f1_values),
        delta_f1_median=statistics.median(delta_f1_values),
        delta_f1_ci=bootstrap_ci_over_graphs(delta_f1_values, seed=43),
        mean_jaccard=statistics.fmean(g.mean_pairwise_jaccard for g in graph_summaries),
        mean_exact_match_rate=statistics.fmean(g.exact_match_rate for g in graph_summaries),
        mean_flip_rate=statistics.fmean(g.flip_rate for g in graph_summaries),
        total_zero_predicted_positive_orderings=sum(g.num_zero_predicted_positive_orderings for g in graph_summaries),
    )
