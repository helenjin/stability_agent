"""Tests for eval_harness/run_experiment.py's scorer-construction wiring.

Focus: `llm_judge_whole` is the one method whose vendored scorer class needs
the raw LLM object (with a `.generate()` method) instead of the wrapped
`EntailmentModel` -- passing the wrong one either breaks at construction
time (unexpected `temperature` kwarg) or at call time (no `.generate()` on
`EntailmentModel`). These tests catch a regression in that routing without
needing any real API calls.
"""
import os
from typing import Dict, List, Tuple

from ares_topodev.eval_harness import _bootstrap  # noqa: F401
from ares_topodev.eval_harness.cache import CachingLLM, DiskPromptCache
from ares_topodev.eval_harness.mock_llm import MockLLM
from ares_topodev.eval_harness.run_experiment import build_scorers, compute_ancestors_by_node_id


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


def _build_dag(steps: Dict[int, str], edges: List[Tuple[int, int]], recipe_name: str = "toy"):
    from ares_topodev.topo_reorder.dag import extract_recipe_dag

    return extract_recipe_dag({"steps": {str(k): v for k, v in steps.items()}, "edges": edges}, recipe_name)


def test_ancestors_equal_direct_parents_for_the_spec_example():
    # START -> A, START -> B, A -> C, B -> D, C -> E, D -> E, E -> END
    # (the SAGER spec's own worked example: A->C, B->D, C,D->E)
    steps = {0: "START", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "END"}
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
    dag = _build_dag(steps, edges)

    ancestors = compute_ancestors_by_node_id(dag)
    assert ancestors[1] == []  # A: root
    assert ancestors[2] == []  # B: root
    assert ancestors[3] == [1]  # C: ancestors == direct parents == {A}
    assert ancestors[4] == [2]  # D: ancestors == direct parents == {B}
    assert ancestors[5] == [1, 2, 3, 4]  # E: Pa_G(E)={C,D}, but Anc_G(E)={A,B,C,D}
    assert ancestors[6] == [1, 2, 3, 4, 5]  # END: every other node


def test_ancestors_transitive_closure_diamond():
    # START -> A -> B -> C, and A -> C directly too (a diamond): Anc_G(C)
    # must include A via BOTH paths, deduplicated, not double-counted or
    # missed because it's already reachable via B.
    steps = {0: "START", 1: "A", 2: "B", 3: "C", 4: "END"}
    edges = [(0, 1), (1, 2), (2, 3), (1, 3), (3, 4)]
    dag = _build_dag(steps, edges)

    ancestors = compute_ancestors_by_node_id(dag)
    assert ancestors[1] == []
    assert ancestors[2] == [1]
    assert ancestors[3] == [1, 2]  # both A and B, not just A (direct) or just B
    assert ancestors[4] == [1, 2, 3]
