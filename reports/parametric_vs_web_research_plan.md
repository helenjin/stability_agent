# Parametric vs Web Knowledge Source Study

## Goal

Study how a model's reasoning changes when its knowledge comes from different sources:
parametric memory, retrieved web evidence, and nonweb retrieved evidence. The central quantity is
not just correctness, but **stability under evidence perturbation**.

For each example:

- there is a task instance such as a claim, question, or hypothesis
- there is a set of evidence units `x`
- there is a base evidence mask `alpha`
- we sample supersets of `alpha` within perturbation radius `r`
- we measure how often the model's judgment stays the same

The resulting stability score asks:

> How invariant is this reasoning outcome to small evidence changes?

This is a robustness or source-sensitivity measure, not a correctness measure by itself.

## Core Research Questions

1. Are judgments based on parametric knowledge more stable than judgments based on retrieved web evidence?
2. When source conditions differ in stability, do they also differ in correctness?
3. On which task types does retrieval improve stability, and on which task types does it introduce instability?
4. When parametric knowledge and retrieved evidence conflict, which source yields more trustworthy behavior?
5. Are low-stability examples driven by noisy evidence, conflicting evidence, sparse support, or brittle reasoning?

## Hypotheses

### H1: Parametric knowledge is often more stable than web retrieval on static domains

Rationale: a closed-book model may produce more internally consistent judgments because its behavior
does not depend on which nearby evidence snippets happen to be included.

### H2: Retrieval helps on dynamic, long-tail, and conflict-heavy tasks

Rationale: web or retrieved evidence should improve the quality of the underlying judgment when the
model's internal memory is stale, incomplete, or contradicted by context.

### H3: Stability and correctness are separable

Rationale: a model can be stably correct, stably wrong, unstable but often correct, or unstable and
wrong. Stability should be reported as a separate axis from quality.

### H4: Source sensitivity is example-dependent, not uniform

Rationale: some examples should be much more stable in the parametric setting, while others should
benefit substantially from retrieval. The interesting cases are the high-gap examples.

## Current Local Result

Balanced comparison over 106 matched MythBusters SciFy/ClaimSpy examples:

| source | n | avg stability | low stability | high stability |
| --- | ---: | ---: | ---: | ---: |
| `ss_rate_dicts_all` | 106 | 0.330 | 60 | 17 |
| `ss_rate_dicts_all_nonweb` | 106 | 0.318 | 72 | 32 |
| `ss_rate_dicts_all_parametric_v0` | 106 | 0.695 | 1 | 36 |

Pairwise source effects:

| comparison | mean delta | median delta | large gaps |
| --- | ---: | ---: | ---: |
| all - parametric | -0.365 | -0.410 | 57 / 106 |
| nonweb - parametric | -0.377 | -0.508 | 80 / 106 |
| all - nonweb | 0.012 | 0.015 | 24 / 106 |

Interpretation: on this local matched set, the parametric condition is much more stable than the
retrieved variants. This does not imply that parametric reasoning is more correct. It means the
output label is less sensitive to evidence-set perturbations.

Existing ClaimSpy score alignment, using the 90 examples with available scores:

| source | n scored | avg score | corr(stability, score) |
| --- | ---: | ---: | ---: |
| `ss_rate_dicts_all` | 90 | 0.325 | 0.031 |
| `ss_rate_dicts_all_nonweb` | 90 | 0.325 | 0.056 |
| `ss_rate_dicts_all_parametric_v0` | 90 | 0.325 | 0.049 |

This suggests stability is measuring a different axis from the existing quality score.

Generated artifacts:

- `reports/source_comparison_balanced.md`
- `reports/source_comparison_balanced.json`

## What Exactly We Would Test

For every dataset and source condition, we would run four complementary analyses.

### 1. Within-source stability

Measure average stability under perturbations when the source type is fixed:

- parametric only
- web retrieved only
- nonweb retrieved only
- hybrid, if available

### 2. Cross-source stability gaps

For the same example, compare:

- `stability(parametric) - stability(web)`
- `stability(parametric) - stability(nonweb)`
- `stability(web) - stability(nonweb)`

This identifies examples that are source-sensitive.

### 3. Stability versus correctness

Jointly evaluate:

- stable and correct
- stable and wrong
- unstable but correct
- unstable and wrong

This is essential for interpreting whether a source is robust in a useful way.

### 4. Error-mode analysis

Audit examples with large source gaps and assign likely causes:

- stale parametric knowledge
- noisy or distracting retrieval
- conflicting evidence
- insufficient support
- multi-hop brittleness
- oversensitivity to added claims

## Dataset-Specific Formulations

### Tier 1: Immediate dataset

**MythBusters SciFy / ClaimSpy**

Why use it:

- already local
- already has `parametric_knowledge`, web-search, and nonweb evidence conditions
- already compatible with `stability_agent`
- already has perturbation outputs and assessment files

Example formulation:

- task: scientific claim assessment
- evidence units `x`: atomic support items or claims
- target: assessment label / score-derived label
- source conditions: `all`, `nonweb`, `parametric`

What to test:

- whether parametric stability remains higher after grouping by domain
- whether high-stability parametric cases are also high-quality
- which examples have the largest parametric-vs-web gaps

### Tier 2: Best external datasets for parametric versus retrieval

**PopQA / EntityQuestions**

Why use it:

- directly framed around parametric versus non-parametric memory
- especially relevant for popularity and long-tail knowledge

Formulation:

- task: short-form question answering
- evidence units `x`: retrieved passages or snippets
- target: answer correctness
- key slice: head versus tail entities

Main test:

- does retrieval lower instability on long-tail questions where parametric memory is weak?

**FreshQA**

Why use it:

- targets fresh, time-sensitive knowledge
- useful for testing stale parametric memory

Formulation:

- task: current-event or current-fact QA
- evidence units `x`: search snippets or retrieved passages
- target: answer correctness
- comparison: closed-book versus web-augmented

Main test:

- does retrieval improve correctness even if it reduces stability, or does it improve both?

**CRAG**

Why use it:

- broad RAG benchmark with diverse sources and metadata
- suitable for popularity, dynamism, and complexity slices

Formulation:

- task: retrieval-augmented QA
- evidence units `x`: web or knowledge-graph evidence chunks
- target: answer correctness or judged factuality

Main test:

- when does broader access to evidence create brittleness versus robustness?

**ConflictBank**

Why use it:

- explicitly targets disagreement between internal and contextual knowledge

Formulation:

- task: answer under conflicting parametric and contextual evidence
- evidence units `x`: contextual claims or passages
- target: whether the model follows parametric memory, contextual evidence, or abstains

Main test:

- how stable is source preference under perturbation when memory and evidence disagree?

### Tier 3: Good stability-metric extensions

**SciFact / SciFact-Open**

- task: scientific claim verification
- evidence units `x`: abstract sentences
- target: SUPPORT / REFUTE / NOINFO
- use case: clean sentence-level perturbations in a scientific setting

**FEVER / KILT**

- task: fact verification or other knowledge-intensive tasks
- evidence units `x`: provenance sentences or passages
- target: label or answer
- use case: standardized large-scale evidence perturbation studies

**ALCE**

- task: attributed long-form QA
- evidence units `x`: cited passages
- target: answer correctness plus citation support
- use case: test whether attribution itself is stable under citation perturbations

## Metric Definitions

### Primary metric

For example `i`, source condition `s`, and radius `r`:

`Stability(i, s, r) = P(model output under perturbed evidence = reference output)`

Aggregate versions:

- per-example average across radii
- source-level average across examples
- pairwise source deltas on matched examples

### Secondary metrics

- correctness or score
- correlation between stability and correctness
- frequency of large cross-source gaps
- disagreement direction, such as more positive or more negative under perturbation
- radius-insensitive disagreement versus knife-edge dependence

### Key reporting table

For each source pair:

- matched example count
- mean stability delta
- median stability delta
- mean absolute gap
- number of examples where left source is clearly more stable
- number where right source is clearly more stable
- number of large-gap examples

## Experimental Design

### Phase 1: Local pilot

1. Use the matched MythBusters / ClaimSpy set.
2. Report source summaries and pairwise source deltas.
3. Slice by domain if metadata is available.
4. Audit top source-sensitive examples manually.

Deliverable:

- one figure showing source-level stability
- one table showing source-pair deltas
- one qualitative section with representative examples

### Phase 2: Correctness disambiguation

1. Join stability outputs with ClaimSpy scores or gold labels.
2. Build a 2x2 categorization:
   - stable-correct
   - stable-wrong
   - unstable-correct
   - unstable-wrong
3. Compare these categories across sources.

Deliverable:

- confusion-style summary of stability versus quality

### Phase 3: External benchmark transfer

Order of expansion:

1. PopQA
2. FreshQA or CRAG
3. ConflictBank
4. SciFact

Rationale:

- PopQA is the cleanest canonical parametric-versus-retrieval benchmark
- FreshQA and CRAG test dynamic and realistic retrieval settings
- ConflictBank isolates source conflict directly
- SciFact tests whether the findings hold in scientific claim verification

### Phase 4: Paper-ready synthesis

Main claims to evaluate:

1. Parametric knowledge is often more stable than retrieved evidence on static tasks.
2. Retrieval is especially valuable for freshness, long-tail knowledge, and conflict resolution.
3. Stability is not equivalent to correctness.
4. Source sensitivity is a useful diagnostic signal for reasoning systems.

## Expected Outcomes

### Plausible positive result

- parametric source is more stable on average
- retrieval improves correctness on freshness and long-tail subsets
- high source-gap examples reveal interpretable failure modes

This would support a nuanced paper claim: internal memory gives consistency, retrieval gives
adaptivity, and stability helps diagnose when each source helps or hurts.

### Plausible negative or mixed result

- retrieval is not systematically less stable
- source effects are mostly task-specific
- correctness tracks source quality more strongly than stability

This would still be publishable if the contribution is framed as a measurement result rather than a
one-directional advantage claim.

## Immediate Next Steps

1. Audit the highest-gap ClaimSpy examples from `source_comparison_balanced.json`.
2. Add a domain-sliced source comparison if domain metadata is recoverable from `problem_id`.
3. Extract representative stable-wrong and unstable-correct examples.
4. Implement the first external dataset adapter, starting with PopQA.
