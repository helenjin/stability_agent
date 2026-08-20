# ARES Valid-Reordering Experiment (TopoDev)

Tests whether ARES and its baselines assign different soundness scores to the
*same* reasoning step when the *same* dependency DAG is presented in different,
but dependency-respecting, linear orders.

See `VENDOR.md` for what was vendored from the official ARES repo and the key
findings that shaped this design (most importantly: **ClaimTrees is out of
scope** -- it's structurally a linear chain with only one valid topological
order in every released config; only **CaptainCookRecipes** has real branching
and thus multiple valid orderings).

## Layout

```
vendor/ares/            unmodified ARES scoring code + CaptainCookRecipes data
vendor/_stubs/           inert import stub for the unused `vllm` dependency
topo_reorder/            DAG extraction (dag.py) + topological sort sampling (topo_sample.py)
eval_harness/             reordering-aware data construction + orchestration + caching
analysis/                 TopoDev + diagnostics + main results table
configs/experiment.yaml   backbone model, K, seed, method list, hyperparameters
results/raw/<method>/<recipe>.json   per-example, per-ordering, per-step raw scores
results/aggregate/*.csv, main_results_table.md
tests/                    topological-sort correctness + claim-text invariance
```

## Setup

```bash
cd /mnt/md0/helenjin/stability_agent
source .venv/bin/activate
pip install -r ares_topodev/requirements.txt
```

## Run the tests (no API calls, no cost)

```bash
python3 -m pytest ares_topodev/tests -q
```

Covers: every sampled ordering satisfies every dependency edge
(`assert_valid_topo_order`); orderings are deduped; END is pinned last;
sampling is reproducible given a seed; an edge-violating swap is correctly
rejected; our reimplementation of the CaptainCookRecipes claim-construction
logic matches the vendored `RecipeGraphDatasetRaw3` byte-for-byte for
derived-claim text and error labels; claim text is provably identical for a
given node id across every sampled ordering; the prompt cache never conflates
two different orderings' prompts.

## Structural dry run (mock LLM, no API calls, no cost)

```bash
python3 -m ares_topodev.eval_harness.run_experiment --config ares_topodev/configs/experiment.yaml --limit 2 --dry-run
python3 -m ares_topodev.analysis.report --config ares_topodev/configs/experiment.yaml
cat ares_topodev/results/aggregate/main_results_table.md
```

This exercises the full pipeline (DAG extraction -> topo sampling -> claim
reordering -> unmodified ARES scoring -> node-identity remapping -> TopoDev
aggregation) end to end with a deterministic fake LLM. **These numbers are not
real results** -- delete `results/raw` and `results/aggregate` before a real
run (or just rerun the real command below, which overwrites them).

## Real run (requires your OpenAI API key and your go-ahead on cost)

```bash
export OPENAI_API_KEY=sk-...
python3 -m ares_topodev.eval_harness.run_experiment --config ares_topodev/configs/experiment.yaml --limit 3   # small subset first
# inspect ares_topodev/results/raw/*/*.json by hand, then:
python3 -m ares_topodev.eval_harness.run_experiment --config ares_topodev/configs/experiment.yaml             # all 24 recipes
python3 -m ares_topodev.analysis.report --config ares_topodev/configs/experiment.yaml
cat ares_topodev/results/aggregate/main_results_table.md
```

**Cost note:** ARES's `cert_nonexact` scorer samples many premise-inclusion
combinations per derived claim (that's the method's whole point -- see the
paper's Section 3.2). In the mock dry run, a single ~15-step recipe under 10
orderings issued on the order of 18,000-29,000 LLM calls for ARES alone (see
`--dry-run` stderr output: `cache_misses=...`). Real GPT-4o-mini pricing is
low per call, but budget for tens of thousands of calls across 24 recipes x 10
orderings x 3 methods before running the full sweep. Start with `--limit 3`.

The backbone model is set by `backbone_model:` in `configs/experiment.yaml`
(any key from `exp_helpers.exp_configs.MODEL_CONFIGS`, e.g. `gpt-4o-mini`,
`gpt-4o`) -- change it there, or copy the config to a new file and pass
`--config path/to/that.yaml`, to test another backbone later.

## Output schema

Each `results/raw/<method>/<recipe_name>.json`:

```json
{
  "dataset": "captaincookrecipes",
  "recipe_name": "ramen",
  "method": "ares",
  "method_config_key": "cert_granular_temp0_nonexact",
  "model": "gpt-4o-mini",
  "hyperparams": {"p": 0.95, "temperature": 0.0},
  "seed": 42,
  "num_orderings_requested": 10,
  "num_orderings_used": 8,
  "topo_sampling_exhausted": true,
  "num_valid_orderings_estimate": {
    "value": 3200000.0, "exact": false,
    "lower_bound_unique_found": 3000, "importance_sample_estimate": 3200000.0
  },
  "step_text_by_node_id": {"3": "Because we have completed ... Remove ...", "...": "..."},
  "ground_truth_error_by_node_id": {"3": 0, "11": 1},
  "orderings": [
    {"ordering_index": 0, "topo_order_step_ids": [3, 6, 14, ...], "scores_by_node_id": {"3": 0.82, "...": "..."}}
  ]
}
```

`results/aggregate/`:
- `topodev_per_example.csv` -- dataset, method, recipe_name, topodev, num_orderings_used
- `topodev_summary.csv` -- dataset, method, n_examples, mean, median, std, ci_low_95, ci_high_95 (bootstrap)
- `per_step_deviation.csv` -- which specific steps are most order-sensitive, with ground-truth error label
- `final_conclusion_deviation.csv` -- deviation on the pinned END/final node specifically
- `position_vs_score.csv` -- long-format (node_id, position, score) for regressing score against sequence position
- `num_valid_orderings.csv` -- how many valid orderings exist/were sampled per recipe
- `main_results_table.md` -- the headline table, with ClaimTrees rows marked N/A (see VENDOR.md) and CaptainCookRecipes rows populated only from files actually on disk

## Determinism / stochasticity

- Temperature fixed at 0.0 everywhere (`configs/experiment.yaml: temperature`).
- GPT-4o-mini is not perfectly deterministic at temperature 0 regardless; this
  is residual stochasticity, not something we control away.
- A fixed `seed` drives topo-sort sampling and the deleted-ingredient draw, so
  the *set of examples and orderings evaluated* is reproducible run-to-run.
  The *LLM outputs* for cache misses are not bitwise-guaranteed reproducible.
- The prompt cache (`results/cache/llm_prompt_cache.jsonl`) is keyed on exact
  rendered prompt text + model + temperature, so reruns reuse prior answers
  for identical prompts without conflating different orderings (see
  `eval_harness/cache.py` docstring and `tests/test_cache.py`).
