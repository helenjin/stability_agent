"""Recipe-level k-fold cross-validated threshold selection for Experiment 2
(error-detection sensitivity under valid reordering).

Why CV, not a single pooled threshold: the paper's own procedure (Section 4)
is itself data-driven -- "we first sweep over all the values that occur in
the training split and select the one that maximizes Macro-F1. We repeat
this process in a 5-fold cross-validation where each time we use one fold
for validation and four folds for testing." There is no single fixed
universal threshold reported in the paper to just reuse. Experiment 1's
`analysis/error_recovery.py::choose_threshold` pools ALL recipes and picks
one threshold from that same pool -- fine for a diagnostic, but it means the
threshold is chosen on the exact same data F1 would be computed on. Since F1
is Experiment 2's *primary* metric, that leakage would optimistically bias
every recipe's reported performance. This module instead assigns recipes to
folds and, for each fold, selects the threshold using only the OTHER
folds' pooled (score, label) pairs -- reusing `error_recovery.choose_threshold`
unchanged, just fed a training subset. No recipe's F1 is ever computed with a
threshold that saw that recipe's own data.

Folds are over *recipes* (graphs), not over orderings or individual
(recipe, ordering) cases -- the K sampled orderings of one recipe are
repeated measurements of the same graph, not independent examples, and must
never be split across train/test for the same recipe.
"""
import random
from dataclasses import dataclass
from typing import Dict, List

from ares_topodev.analysis.error_recovery import choose_threshold


def assign_folds(recipe_names: List[str], k: int = 5, seed: int = 42) -> Dict[str, int]:
    """Deterministic, seeded assignment of each recipe to one of k folds."""
    names = sorted(recipe_names)  # sort first so the shuffle is reproducible independent of input order
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    return {name: i % k for i, name in enumerate(shuffled)}


@dataclass
class CvThresholds:
    method: str
    k: int
    fold_assignment: Dict[str, int]        # recipe_name -> fold index
    threshold_by_fold: Dict[int, float]    # fold index -> threshold chosen from the OTHER folds


def compute_cv_thresholds(raw_results: List[dict], k: int = 5, seed: int = 42) -> CvThresholds:
    """raw_results: one method's list of per-recipe result dicts (e.g. from
    analysis.topodev.load_raw_results). Returns one threshold per fold, each
    chosen using only the recipes NOT in that fold."""
    if not raw_results:
        raise ValueError("compute_cv_thresholds called with no results")

    method = raw_results[0]["method"]
    recipe_names = [r["recipe_name"] for r in raw_results]
    fold_assignment = assign_folds(recipe_names, k=k, seed=seed)

    threshold_by_fold: Dict[int, float] = {}
    for fold in range(k):
        train_results = [r for r in raw_results if fold_assignment[r["recipe_name"]] != fold]
        if not train_results:
            raise ValueError(f"fold {fold} has no training recipes -- k={k} too large for {len(raw_results)} recipes")
        threshold_by_fold[fold] = choose_threshold(train_results)

    return CvThresholds(method=method, k=k, fold_assignment=fold_assignment, threshold_by_fold=threshold_by_fold)


def threshold_for_recipe(cv: CvThresholds, recipe_name: str) -> float:
    fold = cv.fold_assignment[recipe_name]
    return cv.threshold_by_fold[fold]
