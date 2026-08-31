"""SAGER (Structure-Aware Guarantees for Evaluating Reasoning) -- known
dependency graph setting.

Implements exactly the algorithm described in the SAGER known-graph spec:
Monte Carlo propagation of binary soundness variables alpha_v over a DAG,
where a derived node's entailment probability is averaged over up to L
distinct valid topological orderings of the graph, restricted each time to
the subset of depth-limited ancestors that turned out "sound" in that Monte
Carlo sample. See the package README for the full write-up.

Nothing here reimplements or depends on any specific LLM/entailment backend;
plug one in via the `entailment_scorer` callable.
"""
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Union

import networkx as nx

from .entailment import CachingEntailmentScorer, EntailmentScorer
from .graph_utils import (
    compute_depth_limited_ancestors,
    get_topological_traversal,
    restrict_order,
    sample_distinct_topological_orders,
)

Node = Hashable


@dataclass
class SagerResult:
    tau: Dict[Node, float]
    diagnostics: Dict[str, Any]
    # Present only when `debug=True`: one entry per Monte Carlo sample,
    # mapping node -> {"p": p_v^(i), "alpha": alpha_v^(i), "A": sorted(A_v^(i)) or None for base nodes}.
    debug_samples: Optional[List[Dict[Node, Dict[str, Any]]]] = None


def _derived_seed(seed: Optional[int], *parts: object) -> int:
    """A seed deterministically derived from (run seed, *parts), used to draw
    each node's Bernoulli sample independently of traversal position or call
    order -- so the result does not depend on which valid topological
    traversal was chosen as the computation device (see `get_topological_traversal`).
    """
    base = 0 if seed is None else seed
    key = "|".join(["sager_known_graph", str(base)] + [str(p) for p in parts])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sager_known_graph(
    G: nx.DiGraph,
    claims: Dict[Node, str],
    base_priors: Dict[Node, float],
    entailment_scorer: EntailmentScorer,
    depth: Union[int, float, None] = math.inf,
    num_soundness_samples: int = 100,
    max_orderings: int = 20,
    seed: Optional[int] = None,
    debug: bool = False,
) -> SagerResult:
    """Run SAGER (known-graph setting).

    Args:
      G: a networkx.DiGraph, must be acyclic.
      claims: node -> claim text, for every node in G.
      base_priors: node -> prior probability p_v in [0, 1], required for
        every node with no ancestors in G (in-degree 0).
      entailment_scorer: callable(premises: list[str], claim: str) -> float.
      depth: ancestor depth d (positive int, or math.inf/None for unlimited).
      num_soundness_samples: N, number of Monte Carlo samples.
      max_orderings: L, the cap on the number of distinct topological
        orderings averaged over.
      seed: random seed; same seed -> bit-identical results.
      debug: if True, also return per-sample (p_v^(i), alpha_v^(i), A_v^(i))
        for every node. Off by default -- O(N * |V|) memory when enabled.

    Returns:
      SagerResult(tau={node: tau_hat_G(node)}, diagnostics={...}, debug_samples=...)
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be acyclic")
    if num_soundness_samples < 1:
        raise ValueError("num_soundness_samples (N) must be >= 1")
    if max_orderings < 1:
        raise ValueError("max_orderings (L) must be >= 1")

    nodes = list(G.nodes)

    missing_claims = [n for n in nodes if n not in claims]
    if missing_claims:
        raise ValueError(f"claims missing for nodes: {missing_claims}")

    roots = [n for n in nodes if G.in_degree(n) == 0]
    missing_priors = [n for n in roots if n not in base_priors]
    if missing_priors:
        raise ValueError(f"base_priors missing for root (no-ancestor) nodes: {missing_priors}")
    for n in roots:
        p = base_priors[n]
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"base prior for node {n!r} must be in [0, 1], got {p}")
    root_set = set(roots)

    # Step 1: structurally relevant upstream claims + orderings to average over.
    anc_sets = {v: compute_depth_limited_ancestors(G, v, depth) for v in nodes}
    # `traversal` takes no seed -- it is a pure function of G's structure
    # (via nx.topological_sort), never of `seed`. Only `orderings`/Pi_G below
    # is seed-dependent (and only actually affects the result when there are
    # more than max_orderings valid topological orders, forcing the
    # randomized-sampling branch instead of full enumeration).
    traversal = get_topological_traversal(G)  # computation device only
    orderings = sample_distinct_topological_orders(G, max_orderings, seed=seed)  # Pi_G
    L_G = len(orderings)

    # pi_ell restricted to Anc_G^(d)(v) does not depend on the Monte Carlo
    # sample, only on the (fixed) graph structure -- precompute once per node
    # and further filter down to A_v^(i) inside the sample loop.
    restricted_by_node = {
        v: [restrict_order(order, anc_sets[v]) for order in orderings] for v in nodes
    }

    cache = CachingEntailmentScorer(entailment_scorer)

    N = num_soundness_samples
    sum_p: Dict[Node, float] = {v: 0.0 for v in nodes}
    debug_samples: Optional[List[Dict[Node, Dict[str, Any]]]] = [] if debug else None

    # i is the Monte Carlo sample index: which of the N independent passes
    # over the graph we're on. Each pass draws its own full assignment of
    # alpha_v^(i) for every node v (one "possible world"); tau_hat_G(v) is
    # then just the average of p_v^(i) over all N passes. i is threaded into
    # _derived_seed(...) below so that pass i draws different-looking samples
    # than pass i+1 for the same node, while still being reproducible for a
    # fixed (seed, node, i).
    for i in range(N):
        alpha: Dict[Node, int] = {}
        sample_debug: Optional[Dict[Node, Dict[str, Any]]] = {} if debug else None

        # Step 2: propagate soundness uncertainty, ancestors before descendants.
        for v in traversal:
            if v in root_set:
                p_v = base_priors[v]
                active_ids = None
            else:
                anc = anc_sets[v]
                active = {u for u in anc if alpha.get(u, 0) == 1}  # A_v^(i)
                scores = []
                for restricted in restricted_by_node[v]:
                    seq = [u for u in restricted if u in active]  # pi_ell|_{A_v^(i)}
                    premise_texts = [claims[u] for u in seq]
                    scores.append(cache(tuple(seq), v, premise_texts, claims[v]))
                p_v = sum(scores) / L_G
                active_ids = sorted(active, key=str)

            # RNG keyed by (seed, node, sample index) -- NOT by traversal
            # position -- so alpha_v^(i) is independent of which valid
            # topological traversal was used to compute it.
            rng_seed = _derived_seed(seed, "alpha", v, i)
            draw = 1 if (rng_seed / (2**64)) < p_v else 0
            alpha[v] = draw

            if debug:
                sample_debug[v] = {"p": p_v, "alpha": draw, "A": active_ids}

            sum_p[v] += p_v

        if debug:
            debug_samples.append(sample_debug)

    # Step 3: graph-conditioned soundness estimate.
    tau = {v: sum_p[v] / N for v in nodes}

    diagnostics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_soundness_samples": N,
        "num_topological_orders_used": L_G,
        "entailment_requests": cache.total_requests,
        "unique_entailment_calls": cache.unique_calls,
        "cache_hits": cache.cache_hits,
    }

    return SagerResult(tau=tau, diagnostics=diagnostics, debug_samples=debug_samples)
