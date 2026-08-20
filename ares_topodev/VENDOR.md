# Vendored ARES source

`vendor/ares/` contains an unmodified copy of a subset of the official ARES
repository, used to faithfully reproduce ARES's scoring behavior rather than
reimplementing it.

- Source: https://github.com/fallcat/ares
- Commit: `21ff889d038c04369d232181c7059c041d693ecb` (2026-06-17)
- License: MIT (c) 2025 Weiqiu You -- see `vendor/ares/LICENSE`

## What was vendored, and why only this subset

- `src/exp_helpers/` -- the entire package (datasets, methods, models, configs).
  This is ARES's scoring implementation: `datasets/base.py:get_data_entry`,
  `methods/*.py` (ARES/Entail-Prev/Entail-Base/ROSCOE/ReCEval/LLM-Judge
  scorers), `methods/loader.py:get_stability_scorer`, and
  `models/entailment_model.py:EntailmentModel`. **Nothing in this directory
  has been edited.**
- `data/recipe_graphs/*.json` -- the 24 CaptainCookRecipes examples (`steps`,
  `edges`, `topo_order`, ingredient metadata). This is the one dataset with an
  explicit, reorderable dependency DAG (see "Key findings" below).

Not vendored: `metrics/` (paper's own analysis notebooks/figures),
`notebooks/` (kept only as source for the exact custom entailment prompt, now
copied into `eval_harness/entailment_prompts.py`), `results/`, and every
non-recipe dataset (ClaimTrees/synthchain, PRMBench, DeltaBench, GridPuzzle,
etc.) -- out of scope per the plan (see "ClaimTrees" below).

## Import-time dependency stubbing (not a modification of ARES's logic)

`exp_helpers/models/__init__.py` -> `models/loader.py` eagerly imports every
backend (`vllm.py`, `qwen_prm_vllm.py`, `phi_llm.py`, `flan_t5_llm.py`, etc.)
even though this experiment only ever uses the OpenAI backend. Real,
lightweight dependencies (`transformers`, `nltk`, `diskcache`, `anthropic`,
`google-genai`, `torchvision`, `pillow`) are installed for real. The one
exception is `vllm` (a large, GPU/CUDA-oriented package with no functional use
here): `vendor/_stubs/vllm.py` provides an inert stub exposing `LLM` and
`SamplingParams` symbols that raise loudly if anything actually tries to use
them. `eval_harness/_bootstrap.py` puts the stub directory ahead of ARES's
`src/` on `sys.path`. This only affects *import resolution*, not scoring logic.

`models/openai_llm.py` also constructs an `openai.OpenAI()` client at *import*
time, which raises if `OPENAI_API_KEY` is unset (current openai SDK behavior).
`_bootstrap.py` sets a placeholder key via `os.environ.setdefault(...)` so
imports/tests/dry-runs work without a real key; a real key you've already set
always takes precedence.

## Key findings from inspecting this code (informing the experiment design)

1. **ARES's own definition is sequence-position-based, not DAG-based.** The
   premise set for derived claim k is *every* preceding claim
   (`raw_claims + derived_claims[:k]`, a literal list prefix) --
   `datasets/base.py:BaseDataset.get_data_entry` builds this directly, and
   `methods/utils/stability_deterministic.py:tree_stability_rate_deterministic`
   consumes it strictly in list order. The `children`/`parents` dicts
   `get_data_entry` also returns are computed from this same prefix logic, not
   from a recipe's true graph edges -- they are not "the DAG."
2. **ClaimTrees (`synthchain.py`, every `task_id` variant) always produces a
   strict linear `derived_claims` chain.** A strict chain has exactly one
   valid topological order, so `Topo_K(G)` is trivially `{identity}` -- there
   is no valid reordering to sample. Confirmed in the paper text too
   ("ClaimTrees is a synthetic dataset in which the reasoning chain reasons
   starts from a state ... and reason[s] all the way to another state").
   **ClaimTrees is therefore out of scope for this experiment** (per-user
   decision); `analysis/report.py` reports it as `N/A` with this reasoning,
   not as an omitted row.
3. **CaptainCookRecipes ships an explicit DAG independent of presentation
   order**: each recipe JSON's `edges` field is the ground truth; `topo_order`
   is merely the one linearization the vendored loader happens to use.
   `RecipeGraphDatasetRaw3` (the `recipe_graph3` config, used by
   `ares_demo_captaincookrecipes.ipynb`) derives each claim's text from its
   *true* graph parents (`tgt2src[nid]`) by name, so claim text is a pure
   function of node id + edges -- provably identical no matter which valid
   ordering presents it (see `ares_topodev/tests/test_recipe_example.py`).
4. Every node has text "START" (in-degree 0) or "END" (out-degree 0) exactly
   once per recipe (verified across all 24 files). `derived_claims` excludes
   START entirely (`topo_order[1:]`) but includes END. `edges` alone does
   *not* force END to be last (several recipes have steps with no path to
   END), so `topo_reorder/topo_sample.py` pins END last by convention --
   a documented modeling choice representing recipe completion, not an
   edge-derived necessity.
5. The demo notebook builds a **custom** CaptainCookRecipes entailment system
   prompt via `entailment_model.create_custom_config(...)` (ingredient- and
   precedent-step-aware), not the generic binary/granular prompt in
   `EntailmentModel.MODE_CONFIGS`. `eval_harness/entailment_prompts.py` copies
   this verbatim so our experiment matches the paper's actual setup.
6. Caching in the vendored code is dormant: `openai_llm.py`'s
   `cached_openai_generate` has its `@cache.memoize()` decorator commented
   out. `eval_harness/cache.py` adds a new, separate disk cache keyed on exact
   rendered prompt text (safe across orderings by construction -- different
   premise prefixes produce different prompt strings).
