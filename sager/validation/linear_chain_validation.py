"""Scratch validation script -- linear chain a -> b -> c sanity check for
SAGER (known dependency graph). Not a committed test; a manual/inspectable
run per the user's request."""
import math

import networkx as nx

from sager import (
    MockEntailmentScorer,
    compute_depth_limited_ancestors,
    get_topological_traversal,
    sager_known_graph,
    sample_distinct_topological_orders,
)

G = nx.DiGraph([("a", "b"), ("b", "c")])
claims = {"a": "Claim A", "b": "Claim B", "c": "Claim C"}

print("=" * 70)
print("1. Valid topological orderings")
print("=" * 70)
for L in (1, 5, 20):
    orders = sample_distinct_topological_orders(G, max_orders=L, seed=0)
    print(f"  L={L}: orders={orders}  (L_G={len(orders)})")
    assert orders == [["a", "b", "c"]], orders

print()
print("=" * 70)
print("2-4. Depth-limited ancestor sets")
print("=" * 70)
expected = {
    1: {"a": set(), "b": {"a"}, "c": {"b"}},
    2: {"a": set(), "b": {"a"}, "c": {"a", "b"}},
    math.inf: {"a": set(), "b": {"a"}, "c": {"a", "b"}},
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
print("5. Computational traversal")
print("=" * 70)
traversal = get_topological_traversal(G)
print(f"  traversal = {traversal}")
assert traversal == ["a", "b", "c"]
print("  OK: ancestors processed before descendants (a before b before c)")

print()
print("=" * 70)
print("6. L budget does not change L_G for a graph with only one ordering")
print("=" * 70)
l_g_values = set()
for L in (1, 2, 5, 20, 100):
    orders = sample_distinct_topological_orders(G, max_orders=L, seed=42)
    l_g_values.add(len(orders))
    print(f"  L={L:>3} -> L_G={len(orders)}")
assert l_g_values == {1}
print("  OK: L_G=1 regardless of L")

print()
print("=" * 70)
print("7. Small SAGER run (N=5, debug=True)")
print("=" * 70)
base_priors = {"a": 0.9}  # placeholder prior for the only base node
result = sager_known_graph(
    G,
    claims=claims,
    base_priors=base_priors,
    entailment_scorer=MockEntailmentScorer(),
    depth=math.inf,
    num_soundness_samples=5,
    max_orderings=20,
    seed=7,
    debug=True,
)

for i, sample in enumerate(result.debug_samples):
    print(f"  -- sample i={i} --")
    for v in ("a", "b", "c"):
        rec = sample[v]
        print(f"     {v}: A_v^(i)={rec['A']}  p_v^(i)={rec['p']:.4f}  alpha_v^(i)={rec['alpha']}")

print()
print("  tau_hat_G:", {k: round(v, 4) for k, v in result.tau.items()})
print("  diagnostics:", result.diagnostics)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  All ancestor sets matched: {all_ancestor_sets_match}")
print(f"  Topological ordering used: {traversal}")
print(f"  L_G=1 regardless of L: {l_g_values == {1}}")
