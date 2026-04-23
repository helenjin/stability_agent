# Parametric vs Web Knowledge Source Study

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

Interpretation: on this current local sample, the parametric condition is much more stable than
the retrieved/web variants. This does not yet mean it is more correct; it means its verdicts are
less sensitive to evidence-set perturbations. The next analysis should jointly compare stability,
accuracy/score, and source type.

Existing ClaimSpy score alignment, using the 90 examples with available scores:

| source | n scored | avg score | corr(stability, score) |
| --- | ---: | ---: | ---: |
| `ss_rate_dicts_all` | 90 | 0.325 | 0.031 |
| `ss_rate_dicts_all_nonweb` | 90 | 0.325 | 0.056 |
| `ss_rate_dicts_all_parametric_v0` | 90 | 0.325 | 0.049 |

This suggests stability is measuring a different axis from the existing quality score. The
paper framing should treat stability as a robustness/source-sensitivity metric, not a correctness
metric by itself.

Generated artifact:

- `reports/source_comparison_balanced.md`
- `reports/source_comparison_balanced.json`

## Dataset Tiers

### Tier 1: Use Immediately

**MythBusters SciFy / ClaimSpy**

- Already local.
- Already has source distinctions such as `parametric_knowledge`, web-search output, and nonweb evidence.
- Already compatible with `stability_agent`.
- Best first dataset for a paper-ready pilot figure/table.

### Tier 2: Best External Datasets For Parametric vs Retrieval

**PopQA / EntityQuestions**

- Directly studies when parametric memory works versus retrieval augmentation.
- Especially useful for entity popularity and long-tail factual knowledge.
- Metric adaptation: `x = retrieved passages`, `alpha = retrieved subset`, target = exact-match or judge-normalized answer correctness.

**FreshQA**

- Tests current/time-sensitive knowledge, where stale parametric memory should fail.
- Metric adaptation: compare closed-book answers to search-augmented answers; perturb retrieved search snippets.

**CRAG**

- Broad RAG benchmark with web/KG-style APIs and metadata for dynamism, popularity, and complexity.
- Metric adaptation: perturb web/KG evidence units and measure answer stability/correctness.

**ConflictBank**

- Directly targets conflicts between model-encoded and retrieved/contextual knowledge.
- Metric adaptation: stability target can be whether the model follows parametric knowledge, retrieved knowledge, or abstains.

### Tier 3: Good Stability-Metric Extensions

**SciFact / SciFact-Open**

- Scientific claim verification with abstracts and rationale evidence.
- Metric adaptation: `x = abstract sentences`, `alpha = rationale/evidence sentences`, target = SUPPORT/REFUTE/NOINFO.

**FEVER / KILT**

- General fact verification and knowledge-intensive tasks with provenance.
- Metric adaptation: perturb Wikipedia evidence sentences/passages.

**ALCE**

- Citation-oriented long-form QA.
- Metric adaptation: target can combine answer correctness and citation support; perturb cited/retrieved passages.

## Next Experiments

1. Audit local ClaimSpy examples where parametric is much more stable than web/nonweb.
2. For those examples, compare stability against `likert_score` / `continuous_score` to separate stable-correct from stable-wrong.
3. Produce a source-effect table grouped by problem domain if domain metadata is available.
4. Add PopQA as the first external benchmark because it is the cleanest published parametric-vs-nonparametric framing.
5. Add FreshQA or CRAG after PopQA to test whether the conclusion flips for dynamic or low-popularity facts.
