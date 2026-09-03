"""Scratch validation script -- serialization-invariance check for SAGER
(known dependency graph). Builds two differently-*serialized* versions of
the exact same reasoning example (same V, E, claims, priors, d, N, L,
scorer, seed) and checks that every intermediate and final quantity is
EXACTLY (not approximately) identical, and hunts for hidden dependence on
insertion/list order anywhere in the pipeline."""
import math

import networkx as nx

from sager import (
    MockEntailmentScorer,
    compute_depth_limited_ancestors,
    get_topological_traversal,
    sager_known_graph,
    sample_distinct_topological_orders,
)

CLAIM_TEXT = {"a": "Claim A", "b": "Claim B", "c": "Claim C"}

print("=" * 70)
print("1. Building two differently-serialized versions of the same example")
print("=" * 70)

# R1: claims/nodes supplied in order [a, b, c]; edges added a->c then b->c.
R1_ORDER = ["a", "b", "c"]
G1 = nx.DiGraph()
G1.add_nodes_from(R1_ORDER)
G1.add_edges_from([("a", "c"), ("b", "c")])
claims_1 = {}
for node in R1_ORDER:
    claims_1[node] = CLAIM_TEXT[node]
base_priors_1 = {}
for node in ["a", "b"]:
    base_priors_1[node] = 0.65

# R2: claims/nodes supplied in order [b, a, c]; edges added b->c then a->c.
R2_ORDER = ["b", "a", "c"]
G2 = nx.DiGraph()
G2.add_nodes_from(R2_ORDER)
G2.add_edges_from([("b", "c"), ("a", "c")])
claims_2 = {}
for node in R2_ORDER:
    claims_2[node] = CLAIM_TEXT[node]
base_priors_2 = {}
for node in ["b", "a"]:
    base_priors_2[node] = 0.65

print(f"  R1 node order supplied: {R1_ORDER}   G1.nodes (raw): {list(G1.nodes)}")
print(f"  R2 node order supplied: {R2_ORDER}   G2.nodes (raw): {list(G2.nodes)}")
print(f"  R1 claims dict order:   {list(claims_1.items())}")
print(f"  R2 claims dict order:   {list(claims_2.items())}")
assert list(G1.nodes) != list(G2.nodes)  # confirm the raw serializations really do differ
assert list(claims_1.items()) != list(claims_2.items())
assert set(G1.edges) == set(G2.edges)  # same abstract DAG
print("  Confirmed: raw insertion order genuinely differs between R1 and R2;")
print("  abstract graph (node/edge SET) is identical.")

SHARED_KWARGS = dict(
    entailment_scorer=MockEntailmentScorer(),
    depth=math.inf,
    num_soundness_samples=50,
    max_orderings=10,
    seed=17,
)

print()
print("=" * 70)
print("2. Depth-limited ancestor sets (must match regardless of serialization)")
print("=" * 70)
anc_match = True
for node in ["a", "b", "c"]:
    a1 = compute_depth_limited_ancestors(G1, node, math.inf)
    a2 = compute_depth_limited_ancestors(G2, node, math.inf)
    ok = a1 == a2
    anc_match &= ok
    print(f"  node {node}: R1={a1}  R2={a2}  {'OK' if ok else 'MISMATCH'}")

print()
print("=" * 70)
print("3. Selected valid topological orderings Pi_G (compare as sets)")
print("=" * 70)
orders_1 = sample_distinct_topological_orders(G1, max_orders=10, seed=17)
orders_2 = sample_distinct_topological_orders(G2, max_orders=10, seed=17)
set_1 = {tuple(o) for o in orders_1}
set_2 = {tuple(o) for o in orders_2}
print(f"  R1 orderings (list form): {orders_1}")
print(f"  R2 orderings (list form): {orders_2}")
orderings_match = set_1 == set_2
print(f"  as sets: R1={set_1}  R2={set_2}  {'OK (identical sets)' if orderings_match else 'MISMATCH'}")

print()
print("=" * 70)
print("4. Computational topological traversal (may legitimately differ in value)")
print("=" * 70)
traversal_1 = get_topological_traversal(G1)
traversal_2 = get_topological_traversal(G2)
print(f"  R1 traversal: {traversal_1}")
print(f"  R2 traversal: {traversal_2}")
traversals_differ = traversal_1 != traversal_2
print(f"  traversals differ in concrete value: {traversals_differ}")
print("  (this is fine -- the traversal is only a computation device; the spec does")
print("   not require it to be identical across serializations, only that outputs are)")

print()
print("=" * 70)
print("5. Run SAGER on both and compare EXACT equality of every quantity")
print("=" * 70)
result_1 = sager_known_graph(G1, claims=claims_1, base_priors=base_priors_1, debug=True, **SHARED_KWARGS)
result_2 = sager_known_graph(G2, claims=claims_2, base_priors=base_priors_2, debug=True, **SHARED_KWARGS)

samples_exact_match = True
for i in range(len(result_1.debug_samples)):
    s1, s2 = result_1.debug_samples[i], result_2.debug_samples[i]
    for node in ["a", "b", "c"]:
        r1, r2 = s1[node], s2[node]
        alpha_match = r1["alpha"] == r2["alpha"]
        p_match = r1["p"] == r2["p"]  # exact float equality, not isclose
        A_match = r1["A"] == r2["A"]
        samples_exact_match &= alpha_match and p_match and A_match
        if i < 3:  # print first few samples for inspection
            print(f"  i={i} node={node}: R1(A={r1['A']}, p={r1['p']:.6f}, alpha={r1['alpha']})"
                  f"  R2(A={r2['A']}, p={r2['p']:.6f}, alpha={r2['alpha']})"
                  f"  {'MATCH' if (alpha_match and p_match and A_match) else 'MISMATCH'}")

tau_exact_match = result_1.tau == result_2.tau
print()
print(f"  tau R1: {result_1.tau}")
print(f"  tau R2: {result_2.tau}")
print(f"  tau exactly equal (==): {tau_exact_match}")
print(f"  every per-sample (A, p, alpha) exactly equal across all {len(result_1.debug_samples)} samples: {samples_exact_match}")

print()
print("=" * 70)
print("6. Hunting for hidden dependence on input order")
print("=" * 70)
hidden_dependence_found = not (anc_match and orderings_match and samples_exact_match and tau_exact_match)
print(f"  Ancestor sets order-independent: {anc_match}")
print(f"  Pi_G (as sets) order-independent: {orderings_match}")
print(f"  Traversal allowed to differ (device only) -- did differ: {traversals_differ}")
print(f"  alpha_v^(i), p_v^(i), A_v^(i) identical despite differing traversal: {samples_exact_match}")
print(f"  tau_hat_G(v) identical: {tau_exact_match}")
print(f"  Hidden dependence on insertion/list/dict order found: {hidden_dependence_found}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  R1 = {R1_ORDER}, R2 = {R2_ORDER}")
print(f"  tau_R1 == tau_R2 for every node: {tau_exact_match}")
print(f"  Hidden order-dependence found: {hidden_dependence_found}")
print(f"  Serialization-invariance test: {'PASSED' if (tau_exact_match and samples_exact_match and not hidden_dependence_found) else 'FAILED'}")
