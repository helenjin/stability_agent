# SAGER experiments

Real (not toy-graph) experiments comparing SAGER against ARES, on
CaptainCookRecipes examples with a known ground-truth dependency DAG.

## `ares_vs_sager_topodev.py`

The first real experiment: does ARES's score for a claim change when the
same reasoning graph is presented in a different (but equally valid)
topological order, and does SAGER (which conditions on the graph itself,
never the observed serialization) avoid that sensitivity?

Run with:

```
python -m sager.experiments.ares_vs_sager_topodev
```

Reuses existing, unmodified `ares_topodev` infrastructure for everything
graph/dataset-related (`topo_reorder.dag`, `topo_reorder.topo_sample`,
`eval_harness.recipe_example`) and the real vendored ARES scorer
(`exp_helpers.methods.get_stability_scorer("cert_nonexact", ...)` +
`exp_helpers.datasets.base.BaseDataset.get_data_entry`) -- only the new
`sager` package itself and the experiment orchestration/metrics/output code
are new.

Both methods score every individual (premises, claim) pair through the
literal same `exp_helpers.models.entailment_model.EntailmentModel` instance
(wrapping `MockLLM`, the deterministic zero-cost dry-run stand-in, with the
real CaptainCookRecipes system prompt/label mapping) via the
`ARESBackedEntailmentScorer` adapter -- so any score difference can only come
from which premises each framework shows the model, never from the model
itself.

Outputs (written to `sager/experiments/results_ares_vs_sager/`):
- `row_level_results.csv` -- one row per (example, node, serialization, method).
- `aggregate_results.csv` -- one row per (example, node, method): mean score,
  TopoDev/TopoVar across the observed serializations, min/max/swing, verdict
  flip count.

See the script's own docstring for the exact fixed configuration (d, N, L for
SAGER; epsilon/delta/p for ARES; the recipes and R used for this first
small-scale run) and the sanity checks it runs (valid topo orders, claim
identity preserved, gold labels attached correctly, SAGER never seeing the
observed serialization, exact SAGER invariance check, etc).
