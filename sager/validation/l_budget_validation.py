"""Scratch validation script -- topological-order budget L behavior for
SAGER (known dependency graph). a->d, b->d, c->d: 3 mutually independent
sources -> 3! = 6 valid topological orderings."""
import itertools

import networkx as nx

from sager import sample_distinct_topological_orders

G = nx.DiGraph()
G.add_nodes_from(["a", "b", "c", "d"])
G.add_edges_from([("a", "d"), ("b", "d"), ("c", "d")])

ALL_VALID_ORDERINGS = {
    tuple(p) + ("d",) for p in itertools.permutations(["a", "b", "c"])
}
print(f"All valid orderings ({len(ALL_VALID_ORDERINGS)}):")
for o in sorted(ALL_VALID_ORDERINGS):
    print(f"  {list(o)}")
assert len(ALL_VALID_ORDERINGS) == 6


def is_valid_topo_order(order):
    """An order is valid iff every prefix respects G's edges: d must come
    after a, b, and c (the only constraint here)."""
    return tuple(order) in ALL_VALID_ORDERINGS


print()
print("=" * 70)
print("L-budget cases")
print("=" * 70)

results = {}
for L in (1, 3, 6, 10):
    orders = sample_distinct_topological_orders(G, max_orders=L, seed=0)
    results[L] = orders
    L_G = len(orders)
    all_valid = all(is_valid_topo_order(o) for o in orders)
    all_distinct = len({tuple(o) for o in orders}) == len(orders)
    print(f"\n  L={L}")
    print(f"    L_G = {L_G}")
    print(f"    orderings selected: {orders}")
    print(f"    all valid: {all_valid}")
    print(f"    all distinct: {all_distinct}")

    if L <= 6:
        assert L_G == L, f"expected L_G=={L}, got {L_G}"
    else:
        assert L_G == 6, f"expected L_G==6 (min(L, |Topo(G)|)), got {L_G}"
    assert all_valid
    assert all_distinct

print()
print("=" * 70)
print("min(L, |Topo(G)|) check")
print("=" * 70)
for L in (1, 3, 6, 10):
    expected_L_G = min(L, 6)
    actual_L_G = len(results[L])
    ok = expected_L_G == actual_L_G
    print(f"  L={L:>2}: min(L, |Topo(G)|) = min({L},6) = {expected_L_G}   actual L_G = {actual_L_G}   {'OK' if ok else 'MISMATCH'}")
    assert ok

print()
print("=" * 70)
print("Reproducibility: same seed -> same subset (for L < 6); different seed may differ")
print("=" * 70)
for L in (1, 3):
    orders_seed0_a = sample_distinct_topological_orders(G, max_orders=L, seed=99)
    orders_seed0_b = sample_distinct_topological_orders(G, max_orders=L, seed=99)
    orders_seed1 = sample_distinct_topological_orders(G, max_orders=L, seed=100)
    same_seed_match = orders_seed0_a == orders_seed0_b
    print(f"  L={L}: seed=99 (run A) = {orders_seed0_a}")
    print(f"  L={L}: seed=99 (run B) = {orders_seed0_b}   identical to run A: {same_seed_match}")
    print(f"  L={L}: seed=100        = {orders_seed1}")
    assert same_seed_match, "same seed must give identical subset"

# For L=6/10 (exhaustive branch), the seed is irrelevant since ALL orderings
# are returned regardless -- confirm that too, for completeness.
for L in (6, 10):
    o_a = sample_distinct_topological_orders(G, max_orders=L, seed=1)
    o_b = sample_distinct_topological_orders(G, max_orders=L, seed=2)
    print(f"  L={L}: seed=1 -> {len(o_a)} orderings, seed=2 -> {len(o_b)} orderings"
          f"  (sets equal: {set(map(tuple, o_a)) == set(map(tuple, o_b))})")
    assert set(map(tuple, o_a)) == set(map(tuple, o_b)) == ALL_VALID_ORDERINGS

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for L in (1, 3, 6, 10):
    print(f"  L={L:>2} -> L_G={len(results[L])}")
print("  L_G correctly capped at min(L, 6) in every case: True")
print("  No duplicate orderings observed in any case: True")
print("  Seed-based reproducibility (same seed -> same subset) confirmed for L<6: True")
