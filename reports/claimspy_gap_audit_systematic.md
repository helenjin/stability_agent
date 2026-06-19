# ClaimSpy Large-Gap Audit (Systematic)

This audit extends the hand-picked note in `claimspy_gap_audit.md` to the **full population** of large-gap examples. Of 106 matched ClaimSpy examples, **83 have a cross-source stability gap >= 0.30** (the gap is `max_source_stability - min_source_stability` over the three source conditions `all`, `nonweb`, `parametric_v0`).

Source data: `source_comparison_balanced.json` joined to ClaimSpy assessment files via `stability_agent.claimspy_summary.load_claimspy_metadata`. Quality is the ClaimSpy `continuous_score` (fallback `likert_score`). The stability x quality cell is judged on the **parametric** side (stable := parametric avg stability >= 0.80; correct := quality >= 0.67; wrong := quality <= 0.33).

## Headline

- **Parametric is the most stable source in 65 of 83 gap cases** — but its stability is only *moderate* on average (mean 0.688; >=0.80 in 18 cases, 0.50-0.80 in 38, <0.50 in 9). The v1 note's parametric≈1.0 / retrieved≈0.0 picture is the **extreme tail**, not the typical gap case.
- **Retrieval wins in 18 cases** (nonweb most stable in 12, web `all` in 6). Where retrieval wins it is almost always **nonweb reaching ~1.0** — so perfect stability is not unique to parametric memory.
- **Retrieved verdicts shift toward refutation.** Disagreement direction on the retrieved side is `lower` in 47 cases vs `higher` in 31 — added evidence more often makes the model *more skeptical* than more positive.
- **Stability does not track correctness.** Among the 18 parametric-stable gap cases, quality is essentially a coin flip: 9 stable-correct, 7 stable-wrong, 2 stable-mid (mean quality 0.546). This is the cleanest single-source illustration of H3 (stability ≠ quality).

## Signatures (data patterns, not confirmed causes)

| signature | n | meaning |
| --- | ---: | --- |
| `retrieval_oversensitivity` | 45 | parametric best, retrieved collapses to <=0.10, retrieved tagged `radius_insensitive_disagreement` — a single added claim flips the retrieved verdict |
| `parametric_more_stable` | 20 | parametric best but retrieved not fully collapsed |
| `retrieved_more_stable` | 18 | a retrieved source is the most stable (counter-examples to the parametric story) |

> Caveat: these are **signatures in the serialized outputs**, not verified mechanisms. Distinguishing true causes (stale memory, conflicting evidence, distracting retrieval, insufficient support, multi-hop brittleness) requires reading each example's evidence and re-running probes — see Limitations.

## By Domain

| domain | gap cases | mean gap |
| --- | ---: | ---: |
| computational_tools | 29 | 0.658 |
| alloys | 14 | 0.639 |
| unknown | 11 | 0.597 |
| superconductors | 10 | 0.688 |
| modalities | 7 | 0.613 |
| semiconductors | 5 | 0.615 |
| batteries | 4 | 0.521 |
| alloys_sup | 3 | 0.576 |

## Top 12 Largest Gaps

| ex | problem | domain | q | param | all | nonweb | gap | dir | cell |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 13 | alloys_0014 | alloys | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | higher | stable-correct |
| 43 | computational_tools_0019 | computational_tools | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | higher | stable-wrong |
| 49 | computational_tools_0025 | computational_tools | 0.000 | 1.000 | 0.001 | 0.000 | 1.000 | lower | stable-wrong |
| 66 | modalities_0003 | modalities | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | higher | stable-correct |
| 7 | alloys_0008 | alloys | 0.000 | 1.000 | 0.119 | 0.004 | 0.996 | lower | stable-wrong |
| 57 | computational_tools_0033 | computational_tools | 0.250 | 0.975 | 0.010 | 0.000 | 0.975 | lower | stable-wrong |
| 93 | - | - | - | 0.975 | 0.215 | 0.000 | 0.975 | lower | stable-mid |
| 73 | semiconductors_0004 | semiconductors | 0.000 | 0.964 | 0.001 | 0.000 | 0.964 | higher | stable-wrong |
| 85 | superconductors_0011 | superconductors | 0.000 | 0.930 | 0.043 | 0.000 | 0.930 | lower | stable-wrong |
| 34 | computational_tools_0010 | computational_tools | 1.000 | 0.921 | 0.030 | 0.000 | 0.921 | lower | stable-correct |
| 99 | - | - | - | 0.918 | 0.005 | 0.000 | 0.918 | lower | stable-mid |
| 52 | computational_tools_0028 | computational_tools | 1.000 | 0.911 | 0.000 | 0.000 | 0.911 | higher | stable-correct |

## Stable-Wrong Cases (parametric stable, quality <= 0.33)

Parametric reasoning that is perturbation-robust **and** low-quality — rigidity, not reliability. These are the most important cases for the paper's separability claim.

| ex | problem | param | quality | dir | claim |
| ---: | --- | ---: | ---: | --- | --- |
| 43 | computational_tools_0019 | 1.000 | 0.000 | higher | At STP, nitrogen doping of GaAs at Ga sites is more stable than at As s… |
| 49 | computational_tools_0025 | 1.000 | 0.000 | lower | - |
| 7 | alloys_0008 | 1.000 | 0.000 | lower | Claim stating AlBeMet AM162 can be hardened with tungsten to operate at… |
| 57 | computational_tools_0033 | 0.975 | 0.250 | lower | - |
| 73 | semiconductors_0004 | 0.964 | 0.000 | higher | - |
| 85 | superconductors_0011 | 0.930 | 0.000 | lower | - |
| 30 | computational_tools_0006 | 0.855 | 0.000 | lower | - |

## Counter-Examples: Retrieval More Stable

In 18 gap cases a retrieved source (almost always **nonweb**) is the most stable. Some of these are also high quality (e.g. nonweb stable *and* correct), directly limiting any blanket 'parametric is more robust' claim.

| ex | problem | domain | q | param | all | nonweb | best |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 47 | computational_tools_0023 | computational_tools | 0.750 | 0.380 | 0.434 | 1.000 | all_nonweb |
| 10 | alloys_0011 | alloys | 0.000 | 0.406 | 0.532 | 1.000 | all_nonweb |
| 88 | superconductors_0014 | superconductors | 1.000 | 0.464 | 0.498 | 1.000 | all_nonweb |
| 58 | computational_tools_0034 | computational_tools | 0.000 | 0.454 | 0.521 | 0.002 | all |
| 62 | modalities_0001 | modalities | 0.000 | 0.487 | 0.669 | 1.000 | all_nonweb |
| 8 | alloys_0009 | alloys | 0.000 | 0.472 | 0.510 | 0.000 | all |
| 14 | alloys_0015 | alloys | 0.000 | 0.509 | 0.569 | 1.000 | all_nonweb |
| 3 | alloys_0004 | alloys | 1.000 | 0.448 | 0.489 | 0.000 | all |
| 20 | batteries_0001 | batteries | 0.000 | 0.413 | 0.462 | 0.000 | all |
| 71 | semiconductors_0002 | semiconductors | 0.000 | 0.634 | 0.544 | 1.000 | all_nonweb |
| 31 | computational_tools_0007 | computational_tools | 1.000 | 0.550 | 0.574 | 1.000 | all_nonweb |
| 55 | computational_tools_0031 | computational_tools | 1.000 | 0.572 | 0.665 | 1.000 | all_nonweb |
| 23 | batteries_0004 | batteries | 1.000 | 0.698 | 0.592 | 1.000 | all_nonweb |
| 61 | computational_tools_0037 | computational_tools | 0.000 | 0.639 | 0.598 | 1.000 | all_nonweb |
| 64 | modalities_0002 | modalities | 1.000 | 0.383 | 0.550 | 0.179 | all |
| 90 | - | - | - | 0.757 | 0.652 | 1.000 | all_nonweb |
| 81 | superconductors_0007 | superconductors | 0.000 | 0.681 | 0.653 | 1.000 | all_nonweb |
| 33 | computational_tools_0009 | computational_tools | 0.750 | 0.362 | 0.430 | 0.127 | all |

## Limitations

- **Index alignment is assumed.** Examples map to assessment files by sorted index (`load_claimspy_metadata`); 11 gap cases have no matching assessment (quality `-`). These mappings should be spot-checked before the audit feeds a paper claim.
- **Quality label is a proxy.** ClaimSpy `continuous_score` is treated as ground-truth correctness; it is itself a model-assisted judgment.
- **Signatures are not causes.** The error-mode taxonomy in the research plan (stale memory, conflicting/noisy retrieval, sparse support, multi-hop brittleness) cannot be confirmed from serialized labels alone. Confirming them is the natural Phase-2 follow-up.

