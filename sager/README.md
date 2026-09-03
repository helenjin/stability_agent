# SAGER (known dependency graph)

A clean, modular implementation of SAGER (Structure-Aware Guarantees for
Evaluating Reasoning) for the setting where the inferential dependency DAG
`G` is already known -- no graph inference or partial-graph handling here.

This lives in its own top-level `sager/` package (a sibling of
`ares_topodev`, `stability_agent`, etc.), independent of the older,
ARES-tensor-based `ares_topodev.eval_harness.sager` (which reuses ARES's own
`stability_rate_deterministic` machinery and only ever conditions on direct
graph parents, not depth-limited ancestors, and never averages over multiple
topological orderings). That module still backs `ares_topodev`'s
`run_experiment.py` and is left untouched. This package implements the
algorithm as specified from scratch and is not yet wired into any
experiment harness, dataset loaders, or ARES comparison -- that is
deliberately out of scope for this first version.

## Inputs

| Name | Type | Meaning |
|---|---|---|
| `G` | `networkx.DiGraph` | The dependency DAG. Edge `u -> v` means `u` is an inferential dependency of `v`. |
| `claims` | `dict[node, str]` | Claim text for every node in `G`. |
| `base_priors` | `dict[node, float]` | Soundness prior `p_v`, required for every node with **no ancestors at all** in `G` (in-degree 0) -- the "base claims". |
| `entailment_scorer` | `callable(premises: list[str], claim: str) -> float` | Scores an ordered list of premise claims against a target claim. Output is validated/clipped to `[0, 1]`. |
| `depth` | `int \| math.inf \| None` | Ancestor depth `d`. `1` = direct parents only; `math.inf`/`None` = full ancestor closure. |
| `num_soundness_samples` | `int \| None` | `N`, number of Monte Carlo soundness samples. Defaults to 100 if neither this nor `(epsilon, delta)` is given. Mutually exclusive with `(epsilon, delta)`. |
| `max_orderings` | `int` | `L`, cap on the number of distinct topological orderings averaged over. |
| `epsilon`, `delta` | `float \| None` | Tolerance/failure-probability pair. If both given, `N` is *derived* instead of being a raw hyperparameter: `N = ceil(log(2*|V|/delta) / (2*epsilon**2))`, a Hoeffding-style bound giving `\|tau_hat_G(v) - E[tau_hat_G(v)]\| <= epsilon` for every node simultaneously with probability `>= 1 - delta`. This is the same formula and union-bound convention ARES's `cert_nonexact` method uses for its own N -- valid here because `tau_hat_G(v)` is a mean of N samples that are independent across Monte Carlo passes and bounded in `[0, 1]`, exactly Hoeffding's precondition. **Does not** say anything about whether `max_orderings` (`L`) is large enough -- that's a separate source of error (see `sager/experiments/l_convergence_check.py`). |
| `seed` | `int \| None` | Random seed. Same seed -> bit-identical results. |
| `debug` | `bool` | If `True`, also return per-sample `(p_v^(i), alpha_v^(i), A_v^(i))` for every node (off by default -- `O(N * |V|)` memory). |

## Output

`sager_known_graph(...)` returns a `SagerResult`:

```python
SagerResult(
    tau={node: tau_hat_G(node) for node in G},   # graph-conditioned soundness scores, each in [0, 1]
    diagnostics={
        "num_nodes": ..., "num_edges": ...,
        "num_soundness_samples": N, "epsilon": ..., "delta": ...,  # epsilon/delta are None unless that path was used
        "num_topological_orders_used": L_G,
        "entailment_requests": ..., "unique_entailment_calls": ..., "cache_hits": ...,
    },
    debug_samples=None,  # or a list of length N, see above, if debug=True
)
```

## Algorithm flow

1. **Structurally relevant upstream claims** (`graph_utils.py`)
   - `compute_depth_limited_ancestors(G, v, d)` computes `Anc_G^(d)(v)` for every node.
   - `get_topological_traversal(G)` picks **one** valid topological order, used *only* as a computation device so ancestors are processed before descendants.
   - `sample_distinct_topological_orders(G, L, seed)` computes the actual set `Pi_G` of up to `L` distinct valid topological orderings that entailment is averaged over. If the DAG has `<= L` valid orderings, all of them are used (detected cheaply via a bounded probe of `nx.all_topological_sorts`, not full enumeration for large graphs); otherwise `L` distinct orderings are drawn via a seeded randomized Kahn's-algorithm procedure and deduplicated.
   - **These two orderings serve different roles and must not be conflated**: the traversal decides *when* a node is scored; `Pi_G` decides *what premise serializations* are averaged over when scoring it.

2. **Propagate soundness uncertainty** (`algorithm.py`, Monte Carlo loop over `i = 1..N`)
   - Base nodes (no ancestors): `p_v^(i) = p_v` (the prior).
   - Derived nodes: `A_v^(i) = {u in Anc_G^(d)(v) : alpha_u^(i) = 1}`, then
     `p_v^(i) = (1/L_G) * sum_ell E(pi_ell|_{A_v^(i)}, c_v)`, converting each
     restricted node sequence to its ordered claim texts before calling the
     scorer.
   - `alpha_v^(i) ~ Bernoulli(p_v^(i))`, drawn from a seed derived from
     `(seed, node, sample index)` -- **not** from traversal position -- so
     the result does not depend on which of the (possibly several) valid
     traversals was chosen in step 1.

3. **Estimate graph-conditioned soundness**: `tau_hat_G(v) = mean_i p_v^(i)`.

## Caching

`entailment.CachingEntailmentScorer` memoizes calls keyed by
`(tuple(ordered_premise_node_ids), target_node_id)`. This is purely an
optimization -- given a correct `entailment_scorer`, results are identical
with or without it. `diagnostics` reports `entailment_requests`,
`unique_entailment_calls`, and `cache_hits`.

## Edge cases and ambiguities encountered

- **"Base node" is defined by having no ancestors in `G` at all** (in-degree
  0), not by having an empty depth-limited ancestor set. A node can have
  `Anc_G^(d)(v) = {}` for a small `d` while still having ancestors further
  away in `G` -- that node is still "derived" and goes through the
  entailment-averaging branch (with an empty premise list, which is a valid,
  if degenerate, entailment query), not the prior branch. Only true graph
  roots use `base_priors`.
- **RNG design for order-invariance.** The spec requires the traversal
  chosen in step 1 to be "only a computational device." If the Bernoulli
  draws consumed a single shared RNG stream in traversal order, two
  *different* valid traversals would assign the same stream position to
  different nodes, and results would spuriously depend on which traversal
  was picked. To prevent this, every node's draw uses a seed derived from
  `(run seed, node id, Monte Carlo sample index)` only -- never from
  traversal position -- so results are invariant to which valid traversal
  processes the graph.
- **Topological-serialization invariance (required test 4) has a scope
  caveat.** When the number of valid topological orderings of `G` is `<= L`,
  `Pi_G` is the *complete* set of valid orderings, which is identical
  (as a set) regardless of how `G` was constructed (node/edge insertion
  order) -- so `tau` is exactly reproducible across serializations. When
  `G` is large enough that `Pi_G` is capped by the randomized sampling
  branch, the *specific* orderings sampled can differ between two
  differently-constructed-but-structurally-identical graphs, because the
  random walk's tie-breaking depends on networkx's internal node iteration
  order. In that regime, results across serializations agree only up to
  ordinary Monte Carlo/sampling variance, not bit-for-bit. This is a
  documented limitation, not a bug; the required test uses a small DAG
  where full enumeration applies, avoiding this caveat.
- **`sample_distinct_topological_orders` does not sample uniformly** over
  the set of all valid topological orders in the large-graph branch (that
  distribution is expensive to draw from exactly). It draws valid orders via
  random-source-selection Kahn's algorithm, which is a practical, correct
  (every output is a valid ordering), seeded, deduplicated procedure -- but
  not a uniform one. The spec explicitly allows this ("a practical
  randomized procedure", "do not require exact counting").
- **Entailment scorer with an empty premise list.** This occurs whenever a
  derived node's active ancestor set `A_v^(i)` is empty for a given Monte
  Carlo sample (e.g., none of its ancestors sampled "sound" yet, or `d` is
  small). `MockEntailmentScorer` handles this fine; a real LLM-backed scorer
  plugged in later must also handle an empty `premises` list gracefully.
- **Non-associativity of floating point summation** in the entailment
  average `(1/L_G) * sum_ell ...` means summation order over `Pi_G` is not
  guaranteed to be bit-identical if `Pi_G`'s list order differs between two
  runs, even when the *set* of orderings is identical. In practice, with
  the small `L_G` used in tests, sums are effectively invariant to order.

## Tests

See `sager/tests/test_sager_known_graph.py` for the six required
categories: linear chain, branching DAG, diamond DAG (depth-limited
ancestors), topological-serialization invariance, the `L` cap, and
reproducibility.
