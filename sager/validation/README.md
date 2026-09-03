# SAGER validation scripts

Standalone, printable validation scripts used to manually inspect and
sanity-check the SAGER (known dependency graph) implementation on small,
hand-verifiable DAGs. These are distinct from `sager/tests/`, the committed
pytest suite: each script here is meant to be *read* -- it prints its
intermediate quantities (ancestor sets, orderings, per-sample `A_v^{(i)}`,
`p_v^{(i)}`, `alpha_v^{(i)}`, etc.) so the algorithm's behavior can be
inspected step by step, cross-checked against hand computation, and not just
asserted to pass.

Each script is self-contained and runs with:

```
python -m sager.validation.<script_name>
```

or directly via `python sager/validation/<script_name>.py`.

| Script | What it validates |
|---|---|
| `linear_chain_validation.py` | `a -> b -> c`: single topological ordering, depth-limited ancestors at `d=1,2,inf`, traversal order, `L_G=1` regardless of `L`, small debug-traced run. |
| `branching_dag_validation.py` | `a -> c, b -> c`: two distinct topological orderings, `restrict_order` behavior, and an explicit check that entailment averaging over both orderings' distinct premise serializations matches manual computation. |
| `diamond_dag_validation.py` | `a->b, a->c, b->d, c->d`: depth-limited ancestors at `d=1,2,inf`, two orderings, restriction, and manual verification of the ordering-average for node `d`. |
| `diamond_dag_depth_comparison.py` | Same diamond DAG, but runs `d=1`, `d=2`, `d=inf` under the *same seed* side by side, showing `alpha_a/b/c` stay identical while `d`'s structural context (and resulting `p_d`) changes -- and that `d=2` and `d=inf` are identical for this graph's depth. |
| `serialization_invariance_validation.py` | Builds two differently-serialized versions of the same branching-DAG example (different node/edge/claim insertion order) and checks every intermediate and final quantity is *exactly* identical, even when the computational traversal itself differs between the two. |
| `l_budget_validation.py` | `a->d, b->d, c->d` (3! = 6 orderings): confirms `L_G = min(L, \|Topo(G)\|)` for `L=1,3,6,10`, no duplicates, and seed-based reproducibility of the selected subset when `L < 6`. |
| `caching_validation.py` | `a->c, b->c`: runs SAGER with entailment caching enabled vs. disabled (via swapping `CachingEntailmentScorer` for a pass-through counting wrapper) and confirms identical `tau`/per-sample `p_v^{(i)}`/`alpha_v^{(i)}` either way, with caching saving many real scorer evaluations; also confirms the cache key preserves premise order (`[a,b]` and `[b,a]` are distinct entries). |
| `mc_convergence_validation.py` | Checks Monte Carlo convergence properly: variance of `tau_hat_G(c)` shrinks like `O(1/sqrt(N))` across independent seeds, and the converged value matches an independently, analytically computed expectation -- not just "looks stable". |

None of these scripts are wired into CI or `pytest` discovery (no
`test_*.py` naming, no assertions meant to be run unattended as a gate) --
they're meant for manual inspection of the algorithm's behavior. See
`sager/tests/test_sager_known_graph.py` for the actual automated test suite.
