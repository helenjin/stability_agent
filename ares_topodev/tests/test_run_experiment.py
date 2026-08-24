"""Tests for eval_harness/run_experiment.py's scorer-construction wiring.

Focus: `llm_judge_whole` is the one method whose vendored scorer class needs
the raw LLM object (with a `.generate()` method) instead of the wrapped
`EntailmentModel` -- passing the wrong one either breaks at construction
time (unexpected `temperature` kwarg) or at call time (no `.generate()` on
`EntailmentModel`). These tests catch a regression in that routing without
needing any real API calls.
"""
import os

from ares_topodev.eval_harness import _bootstrap  # noqa: F401
from ares_topodev.eval_harness.cache import CachingLLM, DiskPromptCache
from ares_topodev.eval_harness.mock_llm import MockLLM
from ares_topodev.eval_harness.run_experiment import build_scorers


def _build_entailment_model(mock_llm, cache_path):
    from exp_helpers.models import EntailmentModel

    cache = DiskPromptCache(cache_path)
    cached_llm = CachingLLM(mock_llm, cache)
    entailment_model = EntailmentModel(llm=cached_llm, max_new_tokens=100, batch_size=8)
    return entailment_model, cached_llm


def test_build_scorers_passes_raw_llm_to_llm_judge_whole(tmp_path):
    entailment_model, cached_llm = _build_entailment_model(MockLLM(), str(tmp_path / "cache.jsonl"))
    config = {
        "methods_to_run": ["llm_judge_whole"],
        "method_configs": {"llm_judge_whole": "llm_judge_whole_binary"},
        "p": 0.95,
        "temperature": 0.0,
    }
    # LLMJudgeWholeStabilityScorer.__init__ has no `temperature` parameter and
    # no **kwargs catch-all -- if build_scorers injected `temperature` for
    # this method (the bug this test guards against), this call would raise
    # TypeError: unexpected keyword argument 'temperature', not silently
    # construct something wrong.
    scorers, _ = build_scorers(config, entailment_model, cached_llm)

    scorer = scorers["llm_judge_whole"]
    assert scorer.llm is cached_llm  # the raw LLM, not the EntailmentModel wrapper
    assert not hasattr(scorer, "entailment_model")


def test_build_scorers_still_injects_custom_prompt_for_normal_methods(tmp_path):
    entailment_model, cached_llm = _build_entailment_model(MockLLM(), str(tmp_path / "cache2.jsonl"))
    config = {
        "methods_to_run": ["entail_prev"],
        "method_configs": {"entail_prev": "entail_granular"},
        "p": 0.95,
        "temperature": 0.0,
    }
    scorers, _ = build_scorers(config, entailment_model, cached_llm)

    scorer = scorers["entail_prev"]
    assert scorer.entailment_model is entailment_model
    assert scorer.temperature == 0.0
