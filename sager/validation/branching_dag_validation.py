"""Scratch validation script -- branching DAG a->c, b->c sanity check for
SAGER (known dependency graph). Not a committed test; a manual/inspectable
run per the user's request."""
import math

import networkx as nx

from sager import (
    MockEntailmentScorer,
    compute_depth_limited_ancestors,
    get_topological_traversal,
    restrict_order,
    sager_known_graph,
    sample_distinct_topological_orders,
)

G = nx.DiGraph()
G.add_nodes_from(["a", "b", "c"])
G.add_edges_from([("a", "c"), ("b", "c")])
claims = {"a": "Claim A", "b": "Claim B", "c": "Claim C"}
scorer = MockEntailmentScorer()

print("=" * 70)
print("1-2. Valid topological orderings (distinct)")
print("=" * 70)
orders = sample_distinct_topological_orders(G, max_orders=10, seed=0)
order_set = {tuple(o) for o in orders}
print(f"  orderings found: {orders}")
print(f"  as a set: {order_set}")
assert order_set == {("a", "b", "c"), ("b", "a", "c")}
assert len(orders) == len(set(map(tuple, orders)))  # distinct
print("  OK: exactly {[a,b,c], [b,a,c]}, all distinct")

print()
print("=" * 70)
print("3. Anc^(1)_G(c)")
print("=" * 70)
anc1_c = compute_depth_limited_ancestors(G, "c", 1)
print(f"  Anc^(1)(c) = {anc1_c}")
assert anc1_c == {"a", "b"}
print("  OK")

print()
print("=" * 70)
print("4. Restriction pi|_A for A={a,b}")
print("=" * 70)
A = {"a", "b"}
restrictions = {}
for order in orders:
    r = restrict_order(order, A)
    restrictions[tuple(order)] = r
    print(f"  pi={order}  ->  pi|_A={r}")
assert restrictions[("a", "b", "c")] == ["a", "b"]
assert restrictions[("b", "a", "c")] == ["b", "a"]
print("  OK: restrict_order([a,b,c], {a,b}) = [a,b];  restrict_order([b,a,c], {a,b}) = [b,a]")

print()
print("=" * 70)
print("7 (checked early). Computational traversal vs. Pi_G orderings")
print("=" * 70)
traversal = get_topological_traversal(G)
print(f"  computational traversal (used to sequence alpha propagation): {traversal}")
print(f"  Pi_G (used to average entailment):                            {orders}")
print("  These are produced by different functions (get_topological_traversal vs.")
print("  sample_distinct_topological_orders) and serve different roles -- the traversal")
print("  decides computation order only, Pi_G decides premise serializations averaged")
print("  over. They may coincide in value here (small ordering space) without being")
print("  the same mechanism.")

print()
print("=" * 70)
print("5-6. Small SAGER run (N=5, debug=True), manual verification of averaging")
print("=" * 70)
base_priors = {"a": 0.6, "b": 0.6}  # placeholder priors chosen to get a mix of alpha draws
result = sager_known_graph(
    G,
    claims=claims,
    base_priors=base_priors,
    entailment_scorer=scorer,
    depth=1,
    num_soundness_samples=5,
    max_orderings=10,
    seed=3,
    debug=True,
)

# Manually precompute E([a,b], c_c) and E([b,a], c_c) for the explicit check in step 6.
e_ab = scorer([claims["a"], claims["b"]], claims["c"])
e_ba = scorer([claims["b"], claims["a"]], claims["c"])
e_a = scorer([claims["a"]], claims["c"])
e_b = scorer([claims["b"]], claims["c"])
e_empty = scorer([], claims["c"])
print(f"  E([a,b], c_c) = {e_ab:.4f}")
print(f"  E([b,a], c_c) = {e_ba:.4f}")
print(f"  (1/2)[E([a,b],c_c)+E([b,a],c_c)] = {(e_ab + e_ba) / 2:.4f}")
print()

full_average_verified = True
for i, sample in enumerate(result.debug_samples):
    a_rec, b_rec, c_rec = sample["a"], sample["b"], sample["c"]
    A_c = set(c_rec["A"]) if c_rec["A"] is not None else set()
    print(f"  -- sample i={i} --")
    print(f"     alpha_a^(i)={a_rec['alpha']}  alpha_b^(i)={b_rec['alpha']}")
    print(f"     A_c^(i) = {sorted(A_c)}")

    per_order_scores = []
    for order in orders:
        restricted = restrict_order(order, A_c)
        score = scorer([claims[u] for u in restricted], claims["c"])
        per_order_scores.append(score)
        print(f"       pi={order}  pi|_A_c={restricted}  E(pi|_A_c, c_c)={score:.4f}")
    manual_p_c = sum(per_order_scores) / len(orders)
    print(f"     manually averaged p_c^(i) = {manual_p_c:.4f}   SAGER's p_c^(i) = {c_rec['p']:.4f}")
    assert abs(manual_p_c - c_rec["p"]) < 1e-9
    if A_c == {"a", "b"}:
        assert abs(c_rec["p"] - (e_ab + e_ba) / 2) < 1e-9
        print(f"     [A_c={{a,b}}] confirmed p_c^(i) == (1/2)[E([a,b],c_c)+E([b,a],c_c)] = {(e_ab + e_ba) / 2:.4f}")
    print(f"     alpha_c^(i)={c_rec['alpha']}")

print()
print("  tau_hat_G:", {k: round(v, 4) for k, v in result.tau.items()})
print("  diagnostics:", result.diagnostics)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Orderings found: {orders}")
print(f"  Restriction behaved correctly: True")
print(f"  Ordering average verified against manual computation for every sample: True")
