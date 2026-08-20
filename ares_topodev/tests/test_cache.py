import os
import tempfile

from ares_topodev.eval_harness.cache import CachingLLM, DiskPromptCache


class _CountingLLM:
    model_name = "counting-llm"

    def __init__(self):
        self.calls = 0

    def generate(self, prompts, max_new_tokens=500, temperature=0.0, **kwargs):
        self.calls += 1
        return [f"response-to:{p}" for p in prompts]


def test_identical_prompts_hit_cache_and_skip_the_llm():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskPromptCache(os.path.join(tmp, "cache.jsonl"))
        counting = _CountingLLM()
        wrapped = CachingLLM(counting, cache)

        out1 = wrapped.generate(["premises A -> hypothesis B"], temperature=0.0)
        out2 = wrapped.generate(["premises A -> hypothesis B"], temperature=0.0)

        assert out1 == out2
        assert counting.calls == 1  # second call was a pure cache hit
        assert wrapped.hits == 1
        assert wrapped.misses == 1


def test_different_prompts_from_different_orderings_never_collide():
    """The key ordering-safety property: two steps that happen to have
    different overcomplete premise prefixes (i.e. different orderings) must
    never share a cache entry."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskPromptCache(os.path.join(tmp, "cache.jsonl"))
        counting = _CountingLLM()
        wrapped = CachingLLM(counting, cache)

        prompt_ordering_1 = "Context: [claim_A, claim_B]\nHypothesis: claim_C"
        prompt_ordering_2 = "Context: [claim_B, claim_A]\nHypothesis: claim_C"  # same claims, different order

        out1 = wrapped.generate([prompt_ordering_1])
        out2 = wrapped.generate([prompt_ordering_2])

        assert out1 != out2  # distinct prompt text -> distinct (correct) responses
        assert counting.calls == 2
        assert wrapped.hits == 0
        assert wrapped.misses == 2


def test_cache_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.jsonl")
        cache1 = DiskPromptCache(path)
        counting = _CountingLLM()
        wrapped1 = CachingLLM(counting, cache1)
        wrapped1.generate(["hello"])

        cache2 = DiskPromptCache(path)  # simulate a fresh process loading the same cache file
        wrapped2 = CachingLLM(_CountingLLM(), cache2)
        out = wrapped2.generate(["hello"])
        assert out == ["response-to:hello"]
        assert wrapped2.misses == 0
        assert wrapped2.hits == 1
