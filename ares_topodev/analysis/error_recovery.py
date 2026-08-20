"""Error-recovery consistency: does a method flag the *same* ground-truth
error node(s) as unsound regardless of which valid ordering presented them?

This is a different question from TopoDev/TopoVar. Those ask "how much does
the raw score for a fixed claim wobble across orderings?" -- a continuous,
score-level question. This asks the decision-relevant question: once you
threshold a method's scores into a binary "sound / unsound" call (which is
how error detection is actually used downstream), does *reordering alone*
change which nodes get flagged as errors? A method could have modest TopoDev
but still flip its detection decision right across a threshold, or vice
versa -- so this is worth reporting separately, not inferred from TopoDev.

Threshold choice: we pool every (score, ground_truth_label) pair for a method
across all recipes and all sampled orderings in the actual run, then sweep
every distinct observed score value and pick the one maximizing pooled
Macro-F1 against ground truth. This mirrors the paper's own threshold-
selection spirit (Section 4: "sweep over all values that occur ... select the
one that maximizes Macro-F1") without its cross-validation folds -- a
documented simplification, not a reproduction of that exact procedure.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _macro_f1(tp: int, fp: int, fn: int, tn: int) -> float:
    def _f1(tp_, fp_, fn_):
        precision = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        recall = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    f1_error = _f1(tp, fp, fn)
    f1_sound = _f1(tn, fn, fp)  # "sound" class: TN are its true positives, FN/FP swap roles
    return (f1_error + f1_sound) / 2


def choose_threshold(raw_results: List[dict]) -> float:
    """Pools (score, label) pairs across every recipe/ordering/node for one
    method's full result set and returns the score-value threshold (predict
    "error" if score < threshold) maximizing pooled Macro-F1."""
    pairs: List[Tuple[float, int]] = []
    for result in raw_results:
        gt = result["ground_truth_error_by_node_id"]
        for ordering in result["orderings"]:
            for nid, score in ordering["scores_by_node_id"].items():
                pairs.append((score, gt[nid]))

    if not pairs:
        return 0.5

    candidate_thresholds = sorted({score for score, _ in pairs})
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidate_thresholds:
        tp = fp = fn = tn = 0
        for score, label in pairs:
            predicted_error = score < threshold
            if predicted_error and label == 1:
                tp += 1
            elif predicted_error and label == 0:
                fp += 1
            elif not predicted_error and label == 1:
                fn += 1
            else:
                tn += 1
        f1 = _macro_f1(tp, fp, fn, tn)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold


@dataclass
class ErrorRecoveryResult:
    dataset: str
    method: str
    recipe_name: str
    threshold: float
    num_orderings: int
    ground_truth_error_node_ids: List[str]
    predicted_error_node_ids_by_ordering: List[List[str]]
    mean_pairwise_jaccard: float          # how much the PREDICTED error set itself changes across orderings
    frac_orderings_exact_match_gt: float  # how often the predicted set equals ground truth exactly
    mean_recall: float                    # mean, across orderings, of fraction of true error nodes flagged
    mean_precision: float


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_error_recovery(result: dict, threshold: float) -> ErrorRecoveryResult:
    gt = result["ground_truth_error_by_node_id"]
    gt_error_ids = {nid for nid, label in gt.items() if label == 1}

    predicted_sets: List[set] = []
    recalls: List[float] = []
    precisions: List[float] = []

    for ordering in result["orderings"]:
        predicted = {nid for nid, score in ordering["scores_by_node_id"].items() if score < threshold}
        predicted_sets.append(predicted)

        tp = len(predicted & gt_error_ids)
        recalls.append(tp / len(gt_error_ids) if gt_error_ids else 1.0)
        precisions.append(tp / len(predicted) if predicted else (1.0 if not gt_error_ids else 0.0))

    pairwise = []
    for i in range(len(predicted_sets)):
        for j in range(i + 1, len(predicted_sets)):
            pairwise.append(_jaccard(predicted_sets[i], predicted_sets[j]))
    mean_pairwise_jaccard = sum(pairwise) / len(pairwise) if pairwise else 1.0

    exact_matches = sum(1 for p in predicted_sets if p == gt_error_ids)

    return ErrorRecoveryResult(
        dataset=result["dataset"],
        method=result["method"],
        recipe_name=result["recipe_name"],
        threshold=threshold,
        num_orderings=len(predicted_sets),
        ground_truth_error_node_ids=sorted(gt_error_ids),
        predicted_error_node_ids_by_ordering=[sorted(p) for p in predicted_sets],
        mean_pairwise_jaccard=mean_pairwise_jaccard,
        frac_orderings_exact_match_gt=exact_matches / len(predicted_sets) if predicted_sets else 0.0,
        mean_recall=sum(recalls) / len(recalls) if recalls else 0.0,
        mean_precision=sum(precisions) / len(precisions) if precisions else 0.0,
    )
