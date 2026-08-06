# Research Questions

This project studies how a model's reasoning changes when its knowledge comes from
different **sources** — parametric memory, retrieved web evidence, and retrieved
non-web evidence. The central quantity is not correctness alone, but **stability under
evidence perturbation**: how invariant a judgment is to small changes in the evidence
set it was given.

> Stability(i, s, r) = P( model output under perturbed evidence = reference output )
> for example `i`, source condition `s`, perturbation radius `r`.

Stability is treated as a **separate axis from quality** — a model can be stably correct,
stably wrong, unstable but often correct, or unstable and wrong.

## Central question

**Does the source of a model's knowledge change how stable — not just how correct —
its reasoning is, and can that stability signal tell us when each source helps or hurts?**

## Core questions

1. **Source vs. stability.** Are judgments grounded in parametric knowledge more stable
   than judgments grounded in retrieved web evidence (or non-web evidence)?
2. **Stability vs. correctness.** When source conditions differ in stability, do they also
   differ in correctness — or are the two axes separable?
3. **Where retrieval helps vs. hurts.** On which task types does retrieval *improve*
   stability, and on which does it *introduce* instability?
4. **Conflict resolution.** When parametric knowledge and retrieved evidence disagree,
   which source yields more trustworthy behavior under perturbation?
5. **Drivers of instability.** Are low-stability examples driven by noisy evidence,
   conflicting evidence, sparse support, or brittle (e.g. multi-hop) reasoning?

## Working hypotheses

- **H1 — Parametric is often more stable on static domains.** A closed-book model is more
  internally consistent because its output does not depend on which nearby snippets were
  included. (Local pilot supports this: parametric avg stability 0.695 vs ~0.32 for
  retrieved variants on 106 matched MythBusters/ClaimSpy examples.)
- **H2 — Retrieval helps on dynamic, long-tail, and conflict-heavy tasks.** External
  evidence improves the underlying judgment when internal memory is stale, incomplete,
  or contradicted.
- **H3 — Stability and correctness are separable.** They should be reported as distinct
  axes. (Local pilot: correlation between stability and ClaimSpy score ≈ 0.03–0.06.)
- **H4 — Source sensitivity is example-dependent, not uniform.** The interesting cases are
  the high-gap examples — stable under one source, brittle under another.

## How each question is tested

For every dataset and source condition:

1. **Within-source stability** — average stability under perturbation with source fixed
   (parametric / web / non-web / hybrid).
2. **Cross-source stability gaps** — per-example deltas: parametric−web, parametric−non-web,
   web−non-web — to surface source-sensitive examples.
3. **Stability × correctness** — the 2×2: stable-correct, stable-wrong, unstable-correct,
   unstable-wrong, compared across sources.
4. **Error-mode audit** — for large-gap examples, assign a likely cause: stale parametric
   knowledge, noisy/distracting retrieval, conflicting evidence, insufficient support,
   multi-hop brittleness, or oversensitivity to added claims.

## Datasets and what each one probes

- **MythBusters SciFy / ClaimSpy** (local pilot) — scientific claim assessment with existing
  parametric / web / non-web conditions. *Does parametric stability hold up after slicing by
  domain, and are stable parametric cases also high-quality?*
- **PopQA** — entity QA framed around parametric vs. non-parametric memory. *Does retrieval
  reduce instability on long-tail (low-popularity) questions where parametric memory is weak?*
- **ELI5** (long-form QA, `sentence-transformers/eli5`) — open-ended explanatory answers.
  *Is long-form generated reasoning stable under evidence perturbation, and how does that differ
  from short-form factual QA?*
- **FreshQA / CRAG** — time-sensitive and realistic RAG settings. *Does retrieval improve
  correctness even when it reduces stability — or improve both?*
- **ConflictBank** — explicit internal-vs-contextual conflict. *How stable is source preference
  under perturbation when memory and evidence disagree?*
- **SciFact / FEVER / ALCE** — claim verification and attributed QA extensions for testing
  whether findings generalize and whether attribution itself is stable.

## What a result would look like

- **Positive/nuanced:** parametric is more stable on average, retrieval wins on freshness and
  long-tail, and high source-gap examples expose interpretable failure modes — supporting the
  framing *"internal memory gives consistency, retrieval gives adaptivity, and stability
  diagnoses when each helps."*
- **Mixed/negative:** retrieval is not systematically less stable and effects are task-specific
  — still publishable as a **measurement** contribution rather than a one-directional claim.

---
*Distilled from `README.md` and `reports/parametric_vs_web_research_plan.md`. See the plan for
full metric definitions, the phased experimental design, and the current local pilot results.*
