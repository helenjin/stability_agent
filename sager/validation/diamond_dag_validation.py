"""Scratch validation script -- diamond DAG a->b, a->c, b->d, c->d sanity
check for SAGER (known dependency graph). Not a committed test; a
manual/inspectable run, mirroring the linear-chain and branching-DAG
validation scripts."""
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
G.add_nodes_from(["a", "b", "c", "d"])
G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
claims = {"a": "Claim A", "b": "Claim B", "c": "Claim C", "d": "Claim D"}
scorer = MockEntailmentScorer()

print("=" * 70)
print("1. Depth-limited ancestor sets")
print("=" * 70)
expected = {
    1: {"a": set(), "b": {"a"}, "c": {"a"}, "d": {"b", "c"}},
    2: {"a": set(), "b": {"a"}, "c": {"a"}, "d": {"a", "b", "c"}},
    math.inf: {"a": set(), "b": {"a"}, "c": {"a"}, "d": {"a", "b", "c"}},
}
all_ancestor_sets_match = True
for depth, exp_by_node in expected.items():
    for node, exp_set in exp_by_node.items():
        got = compute_depth_limited_ancestors(G, node, depth)
        ok = got == exp_set
        all_ancestor_sets_match &= ok
        depth_label = "inf" if depth == math.inf else depth
        print(f"  d={depth_label:>3}  Anc(G)({node}) = {got}   expected {exp_set}   {'OK' if ok else 'MISMATCH'}")

print()
print("=" * 70)
print("2. Valid topological orderings (a first, d last, b/c interchangeable)")
print("=" * 70)
orders = sample_distinct_topological_orders(G, max_orders=10, seed=0)
order_set = {tuple(o) for o in orders}
print(f"  orderings found: {orders}")
assert order_set == {("a", "b", "c", "d"), ("a", "c", "b", "d")}
assert len(orders) == len(set(map(tuple, orders)))
print("  OK: exactly {[a,b,c,d], [a,c,b,d]}, all distinct")

print()
print("=" * 70)
print("3. Restriction pi|_A for A={b,c} (Anc^(1)(d))")
print("=" * 70)
A = {"b", "c"}
restrictions = {}
for order in orders:
    r = restrict_order(order, A)
    restrictions[tuple(order)] = r
    print(f"  pi={order}  ->  pi|_A={r}")
assert restrictions[("a", "b", "c", "d")] == ["b", "c"]
assert restrictions[("a", "c", "b", "d")] == ["c", "b"]
print("  OK: restrict_order([a,b,c,d], {b,c}) = [b,c];  restrict_order([a,c,b,d], {b,c}) = [c,b]")

print()
print("=" * 70)
print("Computational traversal vs. Pi_G orderings")
print("=" * 70)
traversal = get_topological_traversal(G)
print(f"  computational traversal: {traversal}")
print(f"  Pi_G:                    {orders}")
assert traversal[0] == "a" and traversal[-1] == "d"
print("  OK: traversal processes a before {b,c} before d")

print()
print("=" * 70)
print("4-5. Small SAGER run (N=6, debug=True, depth=inf), manual verification")
print("=" * 70)
base_priors = {"a": 0.7}  # only root
result = sager_known_graph(
    G,
    claims=claims,
    base_priors=base_priors,
    entailment_scorer=scorer,
    depth=math.inf,
    num_soundness_samples=6,
    max_orderings=10,
    seed=11,
    debug=True,
)

for i, sample in enumerate(result.debug_samples):
    print(f"  -- sample i={i} --")
    for v in ("a", "b", "c", "d"):
        rec = sample[v]
        print(f"     {v}: A_v^(i)={rec['A']}  p_v^(i)={rec['p']:.4f}  alpha_v^(i)={rec['alpha']}")

    # Manually recompute p_d^(i) by averaging E(pi|_{A_d}, c_d) over both orderings.
    d_rec = sample["d"]
    A_d = set(d_rec["A"]) if d_rec["A"] is not None else set()
    per_order_scores = []
    for order in orders:
        restricted = restrict_order(order, A_d)
        score = scorer([claims[u] for u in restricted], claims["d"])
        per_order_scores.append((restricted, score))
    manual_p_d = sum(s for _, s in per_order_scores) / len(orders)
    print(f"     manual check for d: {per_order_scores} -> avg={manual_p_d:.4f} (SAGER: {d_rec['p']:.4f})")
    assert abs(manual_p_d - d_rec["p"]) < 1e-9

print()
print("  tau_hat_G:", {k: round(v, 4) for k, v in result.tau.items()})
print("  diagnostics:", result.diagnostics)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  All ancestor sets matched: {all_ancestor_sets_match}")
print(f"  Orderings found: {orders}")
print(f"  Restriction behaved correctly: True")
print(f"  Ordering average verified against manual computation for every sample: True")
