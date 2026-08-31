"""DAG utilities for SAGER (known-dependency-graph setting).

These are pure, structural helpers -- everything here is a function of the
graph `G` (and, for the sampling routine, a seed), never of any text-input
ordering. See the package README for the distinction between:

  * `get_topological_traversal(G)` -- ONE order, used only as a computation
    device so that ancestors are processed before descendants.
  * `sample_distinct_topological_orders(G, ...)` -- the set Pi_G of orderings
    that SAGER actually averages entailment over.

Do not conflate the two.
"""
import itertools
import math
import random
from typing import Hashable, Iterable, List, Optional, Sequence, Set, Union

import networkx as nx

Node = Hashable


def compute_depth_limited_ancestors(
    G: nx.DiGraph, node: Node, depth: Union[int, float, None]
) -> Set[Node]:
    """Anc_G^(d)(node) = {u in V : 1 <= dist_G(u, node) <= depth}.

    Special cases:
      depth == 1                 -> Pa_G(node), direct predecessors only.
      depth in (None, math.inf)  -> Anc_G(node), the full ancestor closure.
    """
    if depth is None:
        depth = math.inf

    if depth == math.inf:
        return set(nx.ancestors(G, node))

    if not isinstance(depth, int) or depth < 1:
        raise ValueError(f"depth must be a positive integer or math.inf/None, got {depth!r}")

    visited: Set[Node] = set()
    frontier: Set[Node] = {node}
    for _ in range(depth):
        next_frontier: Set[Node] = set()
        for n in frontier:
            for pred in G.predecessors(n):
                if pred != node and pred not in visited:
                    next_frontier.add(pred)
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier
    return visited


def get_topological_traversal(G: nx.DiGraph) -> List[Node]:
    """One valid topological order of G, chosen solely to fix a legal
    computation sequence (ancestors before descendants). This is NOT the set
    of orderings averaged over in Step 2 -- see `sample_distinct_topological_orders`.
    """
    return list(nx.topological_sort(G))


def restrict_order(order: Sequence[Node], node_subset: Iterable[Node]) -> List[Node]:
    """pi|_A: the subsequence of `order` containing only nodes in `node_subset`,
    preserving their relative order in `order`."""
    subset = set(node_subset)
    return [n for n in order if n in subset]


def _random_topological_order(G: nx.DiGraph, rng: random.Random) -> List[Node]:
    """One valid topological order drawn via random-source-selection Kahn's
    algorithm: repeatedly pick uniformly at random among the currently
    available (in-degree-zero, among not-yet-emitted nodes) nodes. Not a
    uniform sample over the set of ALL valid topological orders (that
    distribution is expensive to sample from exactly), but every output is a
    valid topological order, and randomness is fully controlled by `rng` --
    the "practical randomized procedure" the spec calls for.
    """
    in_degree = dict(G.in_degree())
    available = [n for n, d in in_degree.items() if d == 0]
    order: List[Node] = []
    while available:
        idx = rng.randrange(len(available))
        available[idx], available[-1] = available[-1], available[idx]
        chosen = available.pop()
        order.append(chosen)
        for succ in G.successors(chosen):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                available.append(succ)
    return order


def sample_distinct_topological_orders(
    G: nx.DiGraph, max_orders: int, seed: Optional[int] = None
) -> List[List[Node]]:
    """Up to `max_orders` DISTINCT valid topological orderings of G.

    If the total number of valid topological orderings is <= max_orders, all
    of them are returned (detected without full enumeration cost via an
    `itertools.islice` probe of at most max_orders + 1 items -- if that probe
    exhausts the generator, the graph's orderings were fully enumerated).
    Otherwise, up to max_orders distinct orderings are drawn with a seeded
    randomized procedure and deduplicated.
    """
    if max_orders < 1:
        raise ValueError("max_orders must be >= 1")

    probe = list(itertools.islice(nx.all_topological_sorts(G), max_orders + 1))
    if len(probe) <= max_orders:
        return [list(order) for order in probe]

    rng = random.Random(seed)
    seen: Set[tuple] = set()
    orders: List[List[Node]] = []
    max_attempts = max(max_orders * 50, 200)
    for _ in range(max_attempts):
        if len(orders) >= max_orders:
            break
        order = _random_topological_order(G, rng)
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            orders.append(order)
    return orders
