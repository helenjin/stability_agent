"""Experiment 2 analysis: for each dependency-violating swap, is the resulting
score change bigger than the ordinary valid-reordering noise Experiment 1
already measured for that same node?

    ViolationEffect_v   = baseline_valid_score_v - invalid_score_v
    NormalizedEffect_v  = |ViolationEffect_v| / TopoDev_v      (TopoDev_v from Experiment 1)

NormalizedEffect_v > 1 means the violation moved the score by more than the
full valid-ordering max-min range already observed for that node -- a signal
that stands out above Experiment 1's noise floor. <= 1 means the violation's
effect is no bigger than ordinary harmless-reordering wobble: the method
can't tell "this reordering is fine" from "this reordering breaks a real
dependency."

`v` is the node whose true prerequisite got displaced (its claim text still
names the displaced node as an already-completed prerequisite, but that node
no longer appears in its premise prefix). `u` is the displaced prerequisite
itself, now evaluated later than its dependents expect.
"""
import json
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

from ares_topodev.analysis.topodev import compute_per_example_topodev, load_raw_results

_ZERO_TOPODEV_EPS = 1e-3
# ^ Below this, TopoDev is treated as "no measurable valid-reordering spread"
# rather than a real denominator. On this [0,1] score scale, a topodev of
# e.g. 1e-7 isn't a legitimately tiny-but-real noise floor -- it's numerically
# indistinguishable from exactly zero, and dividing by it produces enormous,
# meaningless ratios (observed: a single case with topodev ~1.2e-7 produced a
# normalized effect of 508,400x that swamped the mean below). Below this
# threshold we fall back to the same "any nonzero effect exceeds the floor"
# logic used for an exact-zero topodev.


def load_invalid_raw_results(raw_dir: str, method: str) -> List[dict]:
    method_dir = os.path.join(raw_dir, method)
    if not os.path.isdir(method_dir):
        return []
    results = []
    for filename in sorted(os.listdir(method_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(method_dir, filename), "r") as f:
                results.append(json.load(f))
    return results


@dataclass
class CaseEffect:
    recipe_name: str
    method: str
    edge_violated: List[int]
    violation_effect_v: Optional[float]
    normalized_effect_v: Optional[float]      # None if topodev_v ~ 0 (see exceeds_zero_noise_v)
    exceeds_noise_v: Optional[bool]           # |effect| > topodev_v (or > 0 if topodev_v ~ 0)
    correct_direction_v: Optional[bool]       # score dropped (violation_effect_v > 0)
    violation_effect_u: Optional[float]
    normalized_effect_u: Optional[float]
    exceeds_noise_u: Optional[bool]
    correct_direction_u: Optional[bool]


_ZERO_EFFECT_EPS = 1e-9  # threshold for "is the raw score effect itself nonzero" -- unrelated to the topodev-denominator question above


def _node_effect(baseline_score, invalid_score, topodev):
    if baseline_score is None or invalid_score is None:
        return None, None, None, None
    effect = baseline_score - invalid_score
    if topodev is not None and topodev > _ZERO_TOPODEV_EPS:
        normalized = abs(effect) / topodev
        exceeds = normalized > 1.0
    else:
        normalized = None
        exceeds = abs(effect) > _ZERO_EFFECT_EPS  # any nonzero effect exceeds an observed-zero noise floor
    correct_direction = effect > _ZERO_EFFECT_EPS
    return effect, normalized, exceeds, correct_direction


def compute_case_effects(invalid_result: dict, topodev_by_node: Dict[str, float]) -> List[CaseEffect]:
    effects = []
    for case in invalid_result["cases"]:
        u, v = case["edge_violated"]
        effect_v, norm_v, exceeds_v, dir_v = _node_effect(
            case["baseline_valid_score_v"], case["invalid_score_v"], topodev_by_node.get(str(v))
        )
        effect_u, norm_u, exceeds_u, dir_u = _node_effect(
            case["baseline_valid_score_u"], case["invalid_score_u"], topodev_by_node.get(str(u))
        )
        effects.append(
            CaseEffect(
                recipe_name=invalid_result["recipe_name"],
                method=invalid_result["method"],
                edge_violated=[u, v],
                violation_effect_v=effect_v,
                normalized_effect_v=norm_v,
                exceeds_noise_v=exceeds_v,
                correct_direction_v=dir_v,
                violation_effect_u=effect_u,
                normalized_effect_u=norm_u,
                exceeds_noise_u=exceeds_u,
                correct_direction_u=dir_u,
            )
        )
    return effects


def compute_topodev_by_node_per_recipe(baseline_raw_dir: str, method: str) -> Dict[str, Dict[str, float]]:
    """recipe_name -> {node_id: topodev} using Experiment 1's baseline results."""
    baseline_results = load_raw_results(baseline_raw_dir, method)
    out = {}
    for result in baseline_results:
        if not result.get("is_complete", True):
            continue
        pe = compute_per_example_topodev(result)
        out[result["recipe_name"]] = pe.per_step_deviation
    return out


@dataclass
class InvalidSummary:
    method: str
    n_cases: int
    frac_exceeds_noise_v: float
    frac_correct_direction_v: float
    median_normalized_effect_v: Optional[float]  # median, not mean: this ratio has a near-zero-denominator tail (see _ZERO_TOPODEV_EPS)
    frac_exceeds_noise_u: float
    frac_correct_direction_u: float
    median_normalized_effect_u: Optional[float]


def summarize(effects: List[CaseEffect]) -> InvalidSummary:
    if not effects:
        raise ValueError("summarize called with no cases")
    method = effects[0].method

    def _rate(flags):
        flags = [f for f in flags if f is not None]
        return sum(flags) / len(flags) if flags else float("nan")

    def _median_norm(norms):
        norms = [n for n in norms if n is not None]
        return statistics.median(norms) if norms else None

    return InvalidSummary(
        method=method,
        n_cases=len(effects),
        frac_exceeds_noise_v=_rate([e.exceeds_noise_v for e in effects]),
        frac_correct_direction_v=_rate([e.correct_direction_v for e in effects]),
        median_normalized_effect_v=_median_norm([e.normalized_effect_v for e in effects]),
        frac_exceeds_noise_u=_rate([e.exceeds_noise_u for e in effects]),
        frac_correct_direction_u=_rate([e.correct_direction_u for e in effects]),
        median_normalized_effect_u=_median_norm([e.normalized_effect_u for e in effects]),
    )
