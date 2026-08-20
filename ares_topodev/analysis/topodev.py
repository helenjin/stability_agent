"""TopoDev and TopoVar: how much a method's score for the same logical step
varies across valid topological orderings of the same dependency DAG.

    TopoDev_M(G) = (1/|V|) * sum_i [ max_pi tau_{M,i}^pi - min_pi tau_{M,i}^pi ]

TopoDev uses the max-min range per step, which is intuitive but sensitive to a
single outlier ordering. TopoVar is the variance-based companion metric,
averaging each step's *sample variance* across sampled orderings instead of
its range:

    TopoVar_M(G) = (1/|V|) * sum_i Var_pi[ tau_{M,i}^pi ]

Both are computed only over the sampled orderings actually evaluated
(Topo_K(G)), per the plan. TopoVar is more robust to a single extreme ordering
dominating the number (range) but is on a squared-score scale, so the two
metrics can disagree about which method looks "most order-sensitive" when one
method has a single wild outlier vs. another with uniformly-spread scores --
worth reporting both rather than picking one.

Reads raw per-example, per-ordering, per-step JSON files written by
run_experiment.py -- never fabricates a row for an example that wasn't run.
"""
import json
import os
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PerExampleTopoDev:
    dataset: str
    method: str
    recipe_name: str
    topodev: float
    num_orderings_used: int
    per_step_deviation: Dict[str, float]        # node_id -> max-min
    final_conclusion_deviation: Optional[float]  # None if no identifiable final node
    final_node_id: Optional[str]


def load_raw_results(raw_dir: str, method: str) -> List[dict]:
    method_dir = os.path.join(raw_dir, method)
    if not os.path.isdir(method_dir):
        return []
    results = []
    for filename in sorted(os.listdir(method_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(method_dir, filename), "r") as f:
                results.append(json.load(f))
    return results


def compute_per_example_topodev(result: dict) -> PerExampleTopoDev:
    node_ids = list(result["step_text_by_node_id"].keys())
    orderings = result["orderings"]

    per_step_deviation: Dict[str, float] = {}
    for nid in node_ids:
        scores = [o["scores_by_node_id"][nid] for o in orderings if nid in o["scores_by_node_id"]]
        if not scores:
            continue
        per_step_deviation[nid] = max(scores) - min(scores)

    topodev = sum(per_step_deviation.values()) / len(per_step_deviation) if per_step_deviation else 0.0

    # Final conclusion = the node that is last in every ordering (pinned END
    # node for CaptainCookRecipes; see topo_reorder/dag.py). Falls back to
    # None if orderings don't share a common last node (shouldn't happen here).
    final_node_id = None
    last_nodes = {str(o["topo_order_step_ids"][-1]) for o in orderings}
    if len(last_nodes) == 1:
        final_node_id = next(iter(last_nodes))
    final_conclusion_deviation = per_step_deviation.get(final_node_id) if final_node_id else None

    return PerExampleTopoDev(
        dataset=result["dataset"],
        method=result["method"],
        recipe_name=result["recipe_name"],
        topodev=topodev,
        num_orderings_used=result["num_orderings_used"],
        per_step_deviation=per_step_deviation,
        final_conclusion_deviation=final_conclusion_deviation,
        final_node_id=final_node_id,
    )


@dataclass
class PerExampleTopoVar:
    dataset: str
    method: str
    recipe_name: str
    topovar: float
    num_orderings_used: int
    per_step_variance: Dict[str, float]          # node_id -> sample variance across orderings
    final_conclusion_variance: Optional[float]
    final_node_id: Optional[str]


def compute_per_example_topovar(result: dict) -> PerExampleTopoVar:
    node_ids = list(result["step_text_by_node_id"].keys())
    orderings = result["orderings"]

    per_step_variance: Dict[str, float] = {}
    for nid in node_ids:
        scores = [o["scores_by_node_id"][nid] for o in orderings if nid in o["scores_by_node_id"]]
        if len(scores) < 2:
            continue  # sample variance is undefined with fewer than 2 observations
        per_step_variance[nid] = statistics.variance(scores)  # sample variance (ddof=1): orderings are a sample from Topo(G)

    topovar = sum(per_step_variance.values()) / len(per_step_variance) if per_step_variance else 0.0

    final_node_id = None
    last_nodes = {str(o["topo_order_step_ids"][-1]) for o in orderings}
    if len(last_nodes) == 1:
        final_node_id = next(iter(last_nodes))
    final_conclusion_variance = per_step_variance.get(final_node_id) if final_node_id else None

    return PerExampleTopoVar(
        dataset=result["dataset"],
        method=result["method"],
        recipe_name=result["recipe_name"],
        topovar=topovar,
        num_orderings_used=result["num_orderings_used"],
        per_step_variance=per_step_variance,
        final_conclusion_variance=final_conclusion_variance,
        final_node_id=final_node_id,
    )


def bootstrap_ci(values: List[float], num_resamples: int = 10000, seed: int = 42, alpha: float = 0.05):
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
class TopoDevSummary:
    dataset: str
    method: str
    n_examples: int
    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float


def _summarize(dataset: str, method: str, values: List[float]) -> TopoDevSummary:
    ci_low, ci_high = bootstrap_ci(values)
    return TopoDevSummary(
        dataset=dataset,
        method=method,
        n_examples=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        std=statistics.pstdev(values) if len(values) > 1 else 0.0,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def summarize_topodev(per_example: List[PerExampleTopoDev]) -> TopoDevSummary:
    if not per_example:
        raise ValueError("summarize_topodev called with no examples")
    return _summarize(per_example[0].dataset, per_example[0].method, [pe.topodev for pe in per_example])


# Reuses the same summary shape (mean/median/std/bootstrap CI) as TopoDev --
# only the underlying per-example values (topovar instead of topodev) differ.
TopoVarSummary = TopoDevSummary


def summarize_topovar(per_example: List[PerExampleTopoVar]) -> TopoVarSummary:
    if not per_example:
        raise ValueError("summarize_topovar called with no examples")
    return _summarize(per_example[0].dataset, per_example[0].method, [pe.topovar for pe in per_example])
