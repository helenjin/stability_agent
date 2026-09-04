"""Scratch validation script -- entailment caching behavior for SAGER
(known dependency graph). Compares a normal (cached) run against a
no-cache run via `sager_known_graph(..., use_cache=False)`, so the exact
same algorithm code path runs either way -- only the memoization layer
inside CachingEntailmentScorer differs. (Formerly done by monkeypatching
CachingEntailmentScorer out entirely with a hand-rolled pass-through
wrapper -- superseded by the real use_cache flag; see
sager/tests/test_sager_known_graph.py's
test_use_cache_false_disables_memoization_but_preserves_results for the
pytest-suite version of this same comparison.)"""
import math

import networkx as nx

from sager import CachingEntailmentScorer, MockEntailmentScorer, sager_known_graph


class CountingScorer:
    """Wraps the actual entailment scorer with an independent, ground-truth
    counter of real model invocations -- deliberately separate from any
    accounting CachingEntailmentScorer/NoCacheEntailmentScorer do themselves,
    so we're not just trusting the wrapper's own bookkeeping."""

    def __init__(self, scorer):
        self._scorer = scorer
        self.calls = 0

    def __call__(self, premises, claim):
        self.calls += 1
        return self._scorer(premises, claim)


G = nx.DiGraph()
G.add_nodes_from(["a", "b", "c"])
G.add_edges_from([("a", "c"), ("b", "c")])
claims = {"a": "Claim A", "b": "Claim B", "c": "Claim C"}
base_priors = {"a": 0.6, "b": 0.6}
SHARED_KWARGS = dict(
    claims=claims, base_priors=base_priors, depth=math.inf,
    num_soundness_samples=100, max_orderings=10, seed=55,
)

print("=" * 70)
print("1. Run with caching enabled (default)")
print("=" * 70)
counting_scorer_cached = CountingScorer(MockEntailmentScorer())
result_cached = sager_known_graph(G, entailment_scorer=counting_scorer_cached, debug=True, **SHARED_KWARGS)
print(f"  diagnostics: {result_cached.diagnostics}")
print(f"  ground-truth actual scorer invocations: {counting_scorer_cached.calls}")
assert counting_scorer_cached.calls == result_cached.diagnostics["unique_entailment_calls"]

print()
print("=" * 70)
print("2. Run with caching disabled (CachingEntailmentScorer swapped out)")
print("=" * 70)
counting_scorer_nocache = CountingScorer(MockEntailmentScorer())
result_nocache = sager_known_graph(
    G, entailment_scorer=counting_scorer_nocache, debug=True, use_cache=False, **SHARED_KWARGS
)
print(f"  diagnostics: {result_nocache.diagnostics}")
print(f"  ground-truth actual scorer invocations: {counting_scorer_nocache.calls}")
assert counting_scorer_nocache.calls == result_nocache.diagnostics["entailment_requests"]
assert result_nocache.diagnostics["cache_hits"] == 0

print()
print("=" * 70)
print("3. Compare final scores")
print("=" * 70)
tau_match = result_cached.tau == result_nocache.tau
print(f"  tau (cached):    {result_cached.tau}")
print(f"  tau (no-cache):  {result_nocache.tau}")
print(f"  exactly equal: {tau_match}")
assert tau_match

print()
print("=" * 70)
print("4. Compare per-sample p_v^(i) and alpha_v^(i)")
print("=" * 70)
per_sample_match = True
mismatches = []
for i in range(len(result_cached.debug_samples)):
    s_cached, s_nocache = result_cached.debug_samples[i], result_nocache.debug_samples[i]
    for node in ["a", "b", "c"]:
        rc, rn = s_cached[node], s_nocache[node]
        ok = rc["p"] == rn["p"] and rc["alpha"] == rn["alpha"] and rc["A"] == rn["A"]
        per_sample_match &= ok
        if not ok:
            mismatches.append((i, node, rc, rn))
print(f"  all {len(result_cached.debug_samples)} samples x 3 nodes identical (p, alpha, A): {per_sample_match}")
if mismatches:
    print(f"  mismatches: {mismatches[:5]}")
assert per_sample_match

print()
print("=" * 70)
print("5. Entailment evaluations saved by caching")
print("=" * 70)
saved = counting_scorer_nocache.calls - counting_scorer_cached.calls
print(f"  no-cache actual model calls: {counting_scorer_nocache.calls}")
print(f"  cached actual model calls:   {counting_scorer_cached.calls}")
print(f"  evaluations saved by caching: {saved}")
print(f"  cached run's logical requests (both count the same # of *logical* queries): "
      f"{result_cached.diagnostics['entailment_requests']} == {result_nocache.diagnostics['entailment_requests']}: "
      f"{result_cached.diagnostics['entailment_requests'] == result_nocache.diagnostics['entailment_requests']}")
assert result_cached.diagnostics["entailment_requests"] == result_nocache.diagnostics["entailment_requests"]
assert saved > 0
assert saved == result_cached.diagnostics["cache_hits"]

print()
print("=" * 70)
print("6. Cache key preserves premise order ([a,b] != [b,a])")
print("=" * 70)
raw_scorer = MockEntailmentScorer()
probe_cache = CachingEntailmentScorer(raw_scorer)

score_ab = probe_cache(("a", "b"), "c", ["Claim A", "Claim B"], "Claim C")
print(f"  after querying ([a,b], c): unique_calls={probe_cache.unique_calls}  cache_hits={probe_cache.cache_hits}  score={score_ab:.4f}")
assert probe_cache.unique_calls == 1 and probe_cache.cache_hits == 0

score_ba = probe_cache(("b", "a"), "c", ["Claim B", "Claim A"], "Claim C")
print(f"  after querying ([b,a], c): unique_calls={probe_cache.unique_calls}  cache_hits={probe_cache.cache_hits}  score={score_ba:.4f}")
assert probe_cache.unique_calls == 2, "distinct orderings must NOT collapse to the same cache entry"
assert probe_cache.cache_hits == 0, "second distinct-order query must not register as a cache hit"
assert score_ab != score_ba, "MockEntailmentScorer is order-sensitive, so these should differ"

score_ab_again = probe_cache(("a", "b"), "c", ["Claim A", "Claim B"], "Claim C")
print(f"  after re-querying ([a,b], c): unique_calls={probe_cache.unique_calls}  cache_hits={probe_cache.cache_hits}  score={score_ab_again:.4f}")
assert probe_cache.unique_calls == 2, "repeat of an already-seen key must not create a new unique entry"
assert probe_cache.cache_hits == 1, "repeat of an already-seen key must register as a cache hit"
assert score_ab_again == score_ab

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Final tau identical with/without caching: {tau_match}")
print(f"  Per-sample p_v^(i)/alpha_v^(i) identical: {per_sample_match}")
print(f"  Entailment evaluations saved by caching: {saved} (of {result_nocache.diagnostics['entailment_requests']} logical requests)")
print(f"  Cache key correctly distinguishes [a,b] vs [b,a]: True")
all_pass = tau_match and per_sample_match and saved > 0
print(f"  Caching test: {'PASSED' if all_pass else 'FAILED'}")
