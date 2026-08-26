"""Tests for eval_harness/run_experiment.py's scorer-construction wiring.

Focus: `llm_judge_whole` is the one method whose vendored scorer class needs
the raw LLM object (with a `.generate()` method) instead of the wrapped
`EntailmentModel` -- passing the wrong one either breaks at construction
time (unexpected `temperature` kwarg) or at call time (no `.generate()` on
`EntailmentModel`). These tests catch a regression in that routing without
needing any real API calls.
"""
import json
import os
from typing import Dict, List, Tuple

from ares_topodev.eval_harness import _bootstrap  # noqa: F401
from ares_topodev.eval_harness.cache import CachingLLM, DiskPromptCache
from ares_topodev.eval_harness.mock_llm import MockLLM
from ares_topodev.eval_harness.run_experiment import build_scorers, compute_ancestors_by_node_id, run


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


def _base_config(tmp_path, methods_to_run):
    return {
        "dataset": "captaincookrecipes",
        "recipe_data_dir": "ares_topodev/vendor/ares/data/recipe_graphs",
        "backbone_model": "gpt-4o-mini",
        "p": 0.95,
        "temperature": 0.0,
        "max_new_tokens": 100,
        "entailment_batch_size": 8,
        "recipe_concurrency": 1,
        "K": 2,
        "seed": 42,
        "max_topo_attempts": 200,
        "method_configs": {
            "ares": "cert_granular_temp0_nonexact",
            "sager": "sager_gold_graph_reuses_ares_epsilon_delta",
        },
        "methods_to_run": methods_to_run,
        "results_dir": str(tmp_path / "results"),
        "cache_path": str(tmp_path / "cache.jsonl"),
        "raw_claims_shuffle_idx": 0,
    }


def test_already_complete_method_is_not_recomputed_or_rewritten(tmp_path):
    """Regression test: re-running with an already-complete method (e.g.
    ares) alongside a new one (e.g. sager) must skip the complete method
    entirely -- not silently overwrite its file with an in-progress partial
    state from this run's own (even if cache-hit-fast) recomputation. This
    is exactly what happened running sager_ancestors on top of already-
    complete ares/sager Qwen results before this fix."""
    config = _base_config(tmp_path, ["ares"])
    run(config, limit=1, dry_run=True)

    ares_path = os.path.join(config["results_dir"], "raw", "ares", "blenderbananapancakes.json")
    assert os.path.exists(ares_path)
    before = json.load(open(ares_path))
    assert before["is_complete"] is True
    assert before["num_orderings_used"] == 2
    mtime_before = os.path.getmtime(ares_path)

    # Re-run with ares + a NEW method. ares must be left completely untouched.
    config["methods_to_run"] = ["ares", "sager"]
    run(config, limit=1, dry_run=True)

    after = json.load(open(ares_path))
    assert after == before  # byte-identical content, not just equivalent
    assert os.path.getmtime(ares_path) == mtime_before  # never rewritten

    sager_path = os.path.join(config["results_dir"], "raw", "sager", "blenderbananapancakes.json")
    assert os.path.exists(sager_path)
    sager_result = json.load(open(sager_path))
    assert sager_result["is_complete"] is True
    assert sager_result["num_orderings_used"] == 2


def test_partial_existing_file_is_not_skipped(tmp_path):
    """A file marked is_complete=False (or with fewer orderings than
    requested) must still be (re)computed -- the skip only applies to
    genuinely complete results, never to partial ones."""
    config = _base_config(tmp_path, ["ares"])
    ares_dir = os.path.join(config["results_dir"], "raw", "ares")
    os.makedirs(ares_dir, exist_ok=True)
    partial = {
        "dataset": "captaincookrecipes", "recipe_name": "blenderbananapancakes", "method": "ares",
        "is_complete": False, "num_orderings_used": 1, "num_orderings_requested": 2,
        "orderings": [], "failed_orderings": [],
    }
    with open(os.path.join(ares_dir, "blenderbananapancakes.json"), "w") as f:
        json.dump(partial, f)

    run(config, limit=1, dry_run=True)

    result = json.load(open(os.path.join(ares_dir, "blenderbananapancakes.json")))
    assert result["is_complete"] is True
    assert result["num_orderings_used"] == 2  # actually recomputed, not left partial


def test_particle_data_is_opt_in_and_only_for_graph_methods(tmp_path):
    """save_particle_data defaults to off (every existing run/config predates
    this field, and it's real data volume). When enabled, it's only written
    for graph-conditioned methods (sager/sager_ancestors) -- ares has no
    such per-node weighted-sample population to expose."""
    config = _base_config(tmp_path, ["ares", "sager"])
    run(config, limit=1, dry_run=True)

    ares_result = json.load(open(os.path.join(config["results_dir"], "raw", "ares", "blenderbananapancakes.json")))
    sager_result = json.load(open(os.path.join(config["results_dir"], "raw", "sager", "blenderbananapancakes.json")))
    assert "particle_data_by_node_id" not in ares_result["orderings"][0]
    assert "particle_data_by_node_id" not in sager_result["orderings"][0]

    config2 = _base_config(tmp_path, ["ares", "sager"])
    config2["results_dir"] = str(tmp_path / "results2")
    config2["save_particle_data"] = True
    run(config2, limit=1, dry_run=True)

    ares_result2 = json.load(open(os.path.join(config2["results_dir"], "raw", "ares", "blenderbananapancakes.json")))
    sager_result2 = json.load(open(os.path.join(config2["results_dir"], "raw", "sager", "blenderbananapancakes.json")))
    assert "particle_data_by_node_id" not in ares_result2["orderings"][0]  # ares still never gets it
    particle_data = sager_result2["orderings"][0]["particle_data_by_node_id"]
    assert len(particle_data) > 0
    sample_node = next(iter(particle_data.values()))
    assert set(sample_node.keys()) == {"parent_ids", "premises_text", "query_samples", "query_y", "query_counts"}
    assert len(sample_node["query_samples"]) == len(sample_node["query_y"]) == len(sample_node["query_counts"])
    # every query_samples row is as long as premises_text -- one retention
    # bit per premise, so a later analysis can correlate a specific
    # premise's column against query_y
    for row in sample_node["query_samples"]:
        assert len(row) == len(sample_node["premises_text"])
