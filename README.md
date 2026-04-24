# stability_agent

`stability_agent` computes and analyzes soft-stability outputs from the
MythBusters experiments, then surfaces candidate reasoning patterns that
distinguish unstable examples from stable examples.

The package is intentionally local and deterministic: it reads the saved JSON
traces, aggregates stability by radius/source, compares low- and high-stability
examples, and emits heuristic pattern tags for triage.

## Quick Start

From this repository:

```bash
python -m stability_agent.cli \
  ../mythbusters/notebooks/scify_soft_stability/ss_rate_dicts_all \
  ../mythbusters/notebooks/scify_soft_stability/ss_rate_dicts_all_nonweb \
  ../mythbusters/notebooks/scify_soft_stability/ss_rate_dicts_all_parametric \
  --assessment-dir ../mythbusters/data/v2/claimspy_v1_ICL_Agent_2 \
  --markdown-out reports/scify_stability_report.md \
  --json-out reports/scify_stability_report.json
```

## What It Looks For

- Low-vs-high stability groups using configurable thresholds.
- Verdict-shift patterns, such as added claims driving a more negative or more
  positive judgment.
- Radius patterns, such as knife-edge dependencies or radius-insensitive
  disagreement.
- Cross-source sensitivity, such as examples that are unstable only for
  parametric or non-web evidence perturbations.
- Semantic redundancy among support sentences when `--assessment-dir` is given.

Pattern tags are hypotheses for investigation, not causal proof. They are meant
to tell us which examples to open next and what failure mode to inspect.

## Soft-Stability API

The original soft-stability helpers are available inside the package:

```python
from stability_agent import sample_alpha_pertbs, soft_stability_rate
```

These helpers require PyTorch. Install the compute extras when using them:

```bash
pip install -e ".[compute]"
```

## PopQA Workflow

PopQA is a good next dataset for source-sensitive stability because it directly
targets parametric versus retrieved knowledge on entity-centric QA. The base
dataset provides questions, aliases, and popularity metadata; the retrieval/web
condition must be constructed by the experiment.

Once you have:

- PopQA metadata exported locally as `.csv`, `.json`, or `.jsonl`
- stability runs for one or more source conditions

you can join the metadata onto a stability report with:

```python
import json
from pathlib import Path

from stability_agent.popqa_summary import (
    load_popqa_metadata,
    render_popqa_markdown,
    summarize_popqa_source_effects,
)

result = json.loads(Path("reports/popqa_source_comparison.json").read_text())
metadata = load_popqa_metadata(Path("data/popqa.csv"))
summary = summarize_popqa_source_effects(result, metadata)
Path("reports/popqa_followup_summary.md").write_text(
    render_popqa_markdown(summary),
    encoding="utf-8",
)
```

The key PopQA slices to inspect are:

- subject popularity bucket
- object popularity bucket
- pairwise source gaps on matched questions

If the PopQA hypothesis holds, retrieved conditions should help most on
low-popularity questions even if parametric memory remains strong on popular
ones.
