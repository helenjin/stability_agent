"""Entailment scorer interface for SAGER (known-dependency-graph setting).

Kept deliberately decoupled from any specific LLM API: `EntailmentScorer` is
just a callable `(premises: list[str], claim: str) -> float`. Plug in a real
LLM-backed scorer later without touching `algorithm.py`.
"""
import hashlib
import math
from typing import Callable, Dict, Hashable, List, Tuple

Node = Hashable
EntailmentScorer = Callable[[List[str], str], float]


def clip_score(score) -> float:
    """Validate and clip a raw scorer output to [0, 1]."""
    try:
        s = float(score)
    except (TypeError, ValueError) as e:
        raise TypeError(f"entailment scorer must return a number, got {score!r}") from e
    if math.isnan(s):
        raise ValueError("entailment scorer returned NaN")
    return min(1.0, max(0.0, s))


class MockEntailmentScorer:
    """Deterministic, content-hash-based mock entailment scorer for tests and
    examples. The score is a pure function of (premises, claim) text -- no
    randomness, no external state -- so identical inputs always return the
    identical score (this is what the memoization contract in
    `CachingEntailmentScorer` relies on), and no real LLM call is made.

    A larger premise set nudges the score up slightly (`premise_bonus` per
    premise) to loosely emulate "more support -> higher entailment
    probability"; the rest of the score is a hash-derived pseudo-random value
    in [base - spread/2, base + spread/2], clipped to [0, 1].
    """

    def __init__(self, base: float = 0.5, spread: float = 0.3, premise_bonus: float = 0.05):
        self.base = base
        self.spread = spread
        self.premise_bonus = premise_bonus

    def __call__(self, premises: List[str], claim: str) -> float:
        key = "\x1f".join(premises) + "\x1e" + claim
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        frac = int(digest[:8], 16) / 0xFFFFFFFF  # deterministic pseudo-random value in [0, 1]
        score = self.base + (frac - 0.5) * self.spread + self.premise_bonus * len(premises)
        return clip_score(score)


class CachingEntailmentScorer:
    """Memoizing wrapper around an `EntailmentScorer`.

    Cache key is `(tuple(ordered_premise_node_ids), target_node_id)` -- an
    immutable, cheap-to-hash proxy for the actual (premises, claim) text pair
    -- rather than the text itself, per the spec. This is a pure
    optimization: given a correct underlying scorer, results are identical
    with or without this wrapper. `premise_ids` is already the *effective*
    context -- the caller (algorithm.py) has already restricted a sampled
    global topological ordering down to depth-limited ancestors and then
    filtered by which of those turned out "sound" in the current Monte Carlo
    sample -- so two different global orderings that happen to induce the
    same effective context collapse to the same cache entry, while a
    different local order, a different sound/active subset, or a different
    target node all produce a different key. Tracks request/hit counts for
    diagnostics.

    `use_cache=False` disables memoization while keeping the exact same call
    interface and accounting (every request becomes a real model call) --
    this is for A/B numerical-equivalence testing, not a second
    implementation of SAGER: the algorithm code path is identical either
    way, only whether this wrapper remembers past answers differs.
    """

    def __init__(self, scorer: EntailmentScorer, use_cache: bool = True):
        self._scorer = scorer
        self.use_cache = use_cache
        self._cache: Dict[Tuple[tuple, Node], float] = {}
        self.total_requests = 0
        self.cache_hits = 0
        self.model_calls = 0

    def __call__(
        self,
        premise_ids: Tuple[Node, ...],
        target_id: Node,
        premise_texts: List[str],
        claim_text: str,
    ) -> float:
        self.total_requests += 1
        key = (tuple(premise_ids), target_id)
        if self.use_cache and key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.model_calls += 1
        score = clip_score(self._scorer(list(premise_texts), claim_text))
        if self.use_cache:
            self._cache[key] = score
        return score

    @property
    def unique_calls(self) -> int:
        return self.model_calls

    # --- spec-named diagnostics (aliases over the same counters above) -----

    @property
    def num_entailment_requests(self) -> int:
        return self.total_requests

    @property
    def num_entailment_model_calls(self) -> int:
        return self.model_calls

    @property
    def num_cache_hits(self) -> int:
        return self.cache_hits

    @property
    def num_cache_misses(self) -> int:
        return self.total_requests - self.cache_hits

    @property
    def cache_hit_rate(self) -> float:
        return (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0.0
