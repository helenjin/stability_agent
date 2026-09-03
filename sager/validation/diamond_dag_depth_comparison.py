"""Scratch validation script -- diamond DAG a->b, a->c, b->d, c->d, this time
comparing d=1 vs d=2 vs d=inf for the target node d specifically, under a
fixed seed so alpha_a/alpha_b/alpha_c are identical across the three runs
and only d's own structural context changes."""
import math

import networkx as nx

from sager import (
    MockEntailmentScorer,
    compute_depth_limited_ancestors,
    restrict_order,
    sager_known_graph,
    sample_distinct_topological_orders,
)

G = nx.DiGraph()
G.add_nodes_from(["a", "b", "c", "d"])
G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
claims = {"a": "Claim A", "b": "Claim B", "c": "Claim C", "d": "Claim D"}
scorer = MockEntailmentScorer()

print("=" * 70)
print("Ancestor sets")
print("=" * 70)
expected = {
    1: {"a": set(), "b": {"a"}, "c": {"a"}, "d": {"b", "c"}},
    2: {"d": {"a", "b", "c"}},
    math.inf: {"d": {"a", "b", "c"}},
}
all_match = True
for depth, exp_by_node in expected.items():
    for node, exp_set in exp_by_node.items():
        got = compute_depth_limited_ancestors(G, node, depth)
        ok = got == exp_set
        all_match &= ok
        depth_label = "inf" if depth == math.inf else depth
        print(f"  d={depth_label:>3}  Anc(G)({node}) = {got}   expected {exp_set}   {'OK' if ok else 'MISMATCH'}")

print()
print("=" * 70)
print("Valid topological orderings")
print("=" * 70)
orders = sample_distinct_topological_orders(G, max_orders=10, seed=0)
order_set = {tuple(o) for o in orders}
print(f"  orderings found: {orders}")
assert order_set == {("a", "b", "c", "d"), ("a", "c", "b", "d")}
print("  OK: exactly {[a,b,c,d], [a,c,b,d]}")

print()
print("=" * 70)
print("SAGER runs for d = 1, 2, inf  (N=5, same seed=21, same priors)")
print("=" * 70)

SEED = 21
base_priors = {"a": 0.7}
results = {}
for depth in (1, 2, math.inf):
    depth_label = "inf" if depth == math.inf else depth
    result = sager_known_graph(
        G, claims=claims, base_priors=base_priors, entailment_scorer=scorer,
        depth=depth, num_soundness_samples=5, max_orderings=10, seed=SEED, debug=True,
    )
    results[depth] = result

    print(f"\n--- d={depth_label} : Anc^(d)_G(d) = {compute_depth_limited_ancestors(G, 'd', depth)} ---")
    for i, sample in enumerate(result.debug_samples):
        d_rec = sample["d"]
        A_d = set(d_rec["A"]) if d_rec["A"] is not None else set()
        print(f"  sample i={i}: alpha_a={sample['a']['alpha']} alpha_b={sample['b']['alpha']} alpha_c={sample['c']['alpha']}"
              f"  ->  A_d^(i)={sorted(A_d)}")
        per_order = []
        for order in orders:
            restricted = restrict_order(order, A_d)
            score = scorer([claims[u] for u in restricted], claims["d"])
            per_order.append((order, restricted, score))
            print(f"       pi={order}  pi|_A_d={restricted}  E={score:.4f}")
        manual_p_d = sum(s for _, _, s in per_order) / len(orders)
        print(f"       manual avg={manual_p_d:.4f}   SAGER p_d^(i)={d_rec['p']:.4f}   alpha_d^(i)={d_rec['alpha']}")
        assert abs(manual_p_d - d_rec["p"]) < 1e-9

    print(f"  tau_hat_G(d) for d={depth_label}: {results[depth].tau['d']:.4f}")

print()
print("=" * 70)
print("Cross-depth comparison: alpha_a/b/c identical, A_d and p_d differ")
print("=" * 70)
r1, r2, rinf = results[1], results[2], results[math.inf]
for i in range(5):
    a1, a2, ainf = r1.debug_samples[i]["a"], r2.debug_samples[i]["a"], rinf.debug_samples[i]["a"]
    b1, b2, binf = r1.debug_samples[i]["b"], r2.debug_samples[i]["b"], rinf.debug_samples[i]["b"]
    c1, c2, cinf = r1.debug_samples[i]["c"], r2.debug_samples[i]["c"], rinf.debug_samples[i]["c"]
    assert a1["alpha"] == a2["alpha"] == ainf["alpha"]
    assert b1["alpha"] == b2["alpha"] == binf["alpha"]
    assert c1["alpha"] == c2["alpha"] == cinf["alpha"]
d1_A = [set(r1.debug_samples[i]["d"]["A"] or []) for i in range(5)]
d2_A = [set(r2.debug_samples[i]["d"]["A"] or []) for i in range(5)]
dinf_A = [set(rinf.debug_samples[i]["d"]["A"] or []) for i in range(5)]
for i in range(5):
    assert d1_A[i] <= {"b", "c"}
    assert d2_A[i] == dinf_A[i]  # d=2 and d=inf identical for this graph
print("  OK: alpha_a/alpha_b/alpha_c identical across all three depth runs (same seed)")
print("  OK: A_d^(i) at d=1 always subset of {b,c}; d=2 and d=inf give identical A_d^(i) every sample")
print(f"  d=1   A_d per sample: {d1_A}")
print(f"  d=2   A_d per sample: {d2_A}")
print(f"  d=inf A_d per sample: {dinf_A}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  All ancestor sets matched: {all_match}")
print(f"  Orderings: {orders}")
print(f"  tau_hat_G(d): d=1 -> {r1.tau['d']:.4f}, d=2 -> {r2.tau['d']:.4f}, d=inf -> {rinf.tau['d']:.4f}")
