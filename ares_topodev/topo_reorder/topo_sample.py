"""Sample K valid topological orderings of a RecipeDag's derived-claim nodes.

Sampling method (see plan / conversation): randomized Kahn's algorithm run
repeatedly with a seeded RNG.

  1. Compute in-degree for every "middle" node (derived_node_ids minus END,
     which is pinned last -- see dag.py docstring for why).
  2. Maintain the "ready" set = middle nodes with in-degree 0 among those not
     yet placed.
  3. At each step, pick uniformly at random (seeded) from the ready set,
     instead of a fixed tie-break rule -- this is what makes repeated draws
     produce *different* valid orderings instead of the same one every time.
  4. Place that node, decrement in-degree of its successors, add newly-ready
     nodes, repeat until all middle nodes are placed, then append END.
  5. Every drawn ordering is checked against the full edge list before being
     accepted (`assert_valid_topo_order`). Orderings are deduped by their exact
     sequence of step ids. We keep drawing (new seed each attempt) until we
     have K unique orderings or hit `max_attempts`, whichever comes first --
     satisfying "K=10, or all possible topological sorts if fewer than 10
     exist" without ever claiming K when the DAG can't support it.

Also provides an exact/estimated count of the total number of valid
topological sorts (linear extensions), separate from the K we sample.
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .dag import RecipeDag


def assert_valid_topo_order(dag: RecipeDag, order: List[int]) -> None:
    """Raise AssertionError if `order` violates any dependency edge in dag.edges.

    `order` must be exactly a permutation of dag.derived_node_ids (i.e. every
    node except START). START is treated as always occurring before position 0
    (it's never a derived claim); every edge in dag.edges is checked, so this
    also verifies END is compatible with its position (trivially true since we
    always place it last).
    """
    if sorted(order) != sorted(dag.derived_node_ids):
        raise AssertionError(
            f"{dag.recipe_name}: order is not a permutation of derived_node_ids "
            f"(got {len(order)} nodes, expected {len(dag.derived_node_ids)})"
        )
    position = {nid: i for i, nid in enumerate(order)}
    position[dag.start_id] = -1  # START is always before every derived claim
    for (u, v) in dag.edges:
        if position[u] >= position[v]:
            raise AssertionError(
                f"{dag.recipe_name}: edge ({u}->{v}) violated: "
                f"position[{u}]={position[u]} >= position[{v}]={position[v]}"
            )


def _randomized_kahn_order(dag: RecipeDag, rng: random.Random) -> List[int]:
    middle_nodes = [nid for nid in dag.derived_node_ids if nid != dag.end_id]
    middle_edges = [
        (u, v) for (u, v) in dag.sortable_edges() if u != dag.end_id and v != dag.end_id
    ]

    indegree: Dict[int, int] = {nid: 0 for nid in middle_nodes}
    successors: Dict[int, List[int]] = {nid: [] for nid in middle_nodes}
    for (u, v) in middle_edges:
        indegree[v] += 1
        successors[u].append(v)

    ready = sorted(nid for nid, d in indegree.items() if d == 0)
    placed: List[int] = []
    remaining_indegree = dict(indegree)

    while ready:
        pick = rng.choice(ready)
        ready.remove(pick)
        placed.append(pick)
        for succ in successors[pick]:
            remaining_indegree[succ] -= 1
            if remaining_indegree[succ] == 0:
                ready.append(succ)
        ready.sort()  # deterministic given the rng draw sequence

    if len(placed) != len(middle_nodes):
        raise AssertionError(
            f"{dag.recipe_name}: Kahn's algorithm stalled with a cycle among "
            f"derived-claim nodes ({len(placed)}/{len(middle_nodes)} placed) -- "
            "the recipe graph is not a DAG once START/END are excluded."
        )

    return placed + [dag.end_id]


@dataclass
class TopoSampleResult:
    orderings: List[List[int]]         # up to K unique valid orderings (lists of step ids)
    num_requested: int
    num_attempts_used: int
    exhausted: bool                    # True if we stopped because attempts ran out, not because we hit K


def sample_orderings(
    dag: RecipeDag,
    k: int = 10,
    seed: int = 42,
    max_attempts: int = 200,
) -> TopoSampleResult:
    seen: set = set()
    orderings: List[List[int]] = []
    attempt = 0
    while len(orderings) < k and attempt < max_attempts:
        rng = random.Random(f"topo_sample|{seed}|{attempt}")
        order = _randomized_kahn_order(dag, rng)
        assert_valid_topo_order(dag, order)
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            orderings.append(order)
        attempt += 1

    return TopoSampleResult(
        orderings=orderings,
        num_requested=k,
        num_attempts_used=attempt,
        exhausted=(attempt >= max_attempts and len(orderings) < k),
    )


# --- Counting / estimating the total number of valid topological sorts ---

_EXACT_DP_NODE_CUTOFF = 16  # 2^16 * 16 ~= 1M ops, fast; above this we estimate instead


def _exact_count_linear_extensions(dag: RecipeDag) -> int:
    """DP over subsets: dp[S] = # of valid arrangements of exactly the nodes in S
    (as a prefix satisfying all edges among S). dp[full set] is the answer.
    Only used when len(middle_nodes) <= _EXACT_DP_NODE_CUTOFF.
    """
    middle_nodes = [nid for nid in dag.derived_node_ids if nid != dag.end_id]
    n = len(middle_nodes)
    idx = {nid: i for i, nid in enumerate(middle_nodes)}
    middle_edges = [
        (u, v) for (u, v) in dag.sortable_edges() if u != dag.end_id and v != dag.end_id
    ]
    preds_mask = [0] * n
    for (u, v) in middle_edges:
        preds_mask[idx[v]] |= 1 << idx[u]

    full = (1 << n) - 1
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1, 1 << n):
        total = 0
        for i in range(n):
            bit = 1 << i
            if not (mask & bit):
                continue
            prev = mask & ~bit
            if preds_mask[i] & ~prev:
                continue  # some predecessor of i is not yet placed in `prev`
            total += dp[prev]
        dp[mask] = total
    return dp[full]


def _importance_sampling_estimate(dag: RecipeDag, seed: int, num_draws: int = 50) -> float:
    """Unbiased-in-expectation estimator: prod(ready-set size at each step) for
    a random draw equals 1/P(that specific ordering), so averaging it over
    many draws estimates the total number of valid orderings.
    """
    middle_nodes = [nid for nid in dag.derived_node_ids if nid != dag.end_id]
    middle_edges = [
        (u, v) for (u, v) in dag.sortable_edges() if u != dag.end_id and v != dag.end_id
    ]
    estimates = []
    for draw in range(num_draws):
        rng = random.Random(f"topo_estimate|{seed}|{draw}")
        indegree = {nid: 0 for nid in middle_nodes}
        successors = {nid: [] for nid in middle_nodes}
        for (u, v) in middle_edges:
            indegree[v] += 1
            successors[u].append(v)
        ready = sorted(nid for nid, d in indegree.items() if d == 0)
        remaining_indegree = dict(indegree)
        log_product = 0.0
        while ready:
            log_product += math.log(len(ready))
            pick = rng.choice(ready)
            ready.remove(pick)
            for succ in successors[pick]:
                remaining_indegree[succ] -= 1
                if remaining_indegree[succ] == 0:
                    ready.append(succ)
            ready.sort()
        estimates.append(math.exp(log_product))
    return sum(estimates) / len(estimates)


def _lower_bound_via_extended_sampling(dag: RecipeDag, seed: int, max_attempts: int = 3000) -> int:
    """A guaranteed lower bound: number of *distinct* valid orderings actually
    observed over many random draws. Counting linear extensions exactly is
    #P-hard in general, and the importance-sampling estimator below can be
    badly biased/high-variance for skewed DAGs (a long near-linear chain with
    one small branch point has most draws land on ready-set size 1, so a
    modest number of draws rarely samples the rare wide points enough to
    estimate their contribution well). This lower bound is comparatively
    boring but never self-contradicts what we actually found.
    """
    seen = set()
    for attempt in range(max_attempts):
        rng = random.Random(f"topo_lower_bound|{seed}|{attempt}")
        order = _randomized_kahn_order(dag, rng)
        seen.add(tuple(order))
    return len(seen)


@dataclass
class LinearExtensionCount:
    value: float
    exact: bool
    lower_bound_unique_found: int   # always populated; a real lower bound in both branches
    importance_sample_estimate: float  # rough point estimate; can be unreliable for skewed DAGs


def count_or_estimate_linear_extensions(dag: RecipeDag, seed: int = 42) -> LinearExtensionCount:
    n_middle = len([nid for nid in dag.derived_node_ids if nid != dag.end_id])
    if n_middle <= _EXACT_DP_NODE_CUTOFF:
        exact_value = float(_exact_count_linear_extensions(dag))
        return LinearExtensionCount(
            value=exact_value,
            exact=True,
            lower_bound_unique_found=int(exact_value),
            importance_sample_estimate=exact_value,
        )
    lower_bound = _lower_bound_via_extended_sampling(dag, seed=seed)
    point_estimate = _importance_sampling_estimate(dag, seed=seed)
    # Never report an "estimate" smaller than orderings we actually observed.
    return LinearExtensionCount(
        value=max(lower_bound, point_estimate),
        exact=False,
        lower_bound_unique_found=lower_bound,
        importance_sample_estimate=point_estimate,
    )
