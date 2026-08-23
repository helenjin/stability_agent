# Experiment 2 Main Table: Error-Detection Sensitivity Under Valid Reordering

Uncertainty is mean +/- bootstrap SD, resampled over graphs (10000 resamples, ARES paper's own +/- display style; see analysis/detection_sensitivity.py's bootstrap_sd_over_graphs).

| Method | Mean F1 ↑ | ΔF1 ↓ | N graphs |
|---|---:|---:|---:|
| ARES | 0.6431 ± 0.0393 | 0.2788 ± 0.0388 | 24 |
| Entail-Prev | 0.3833 ± 0.0266 | 0.2227 ± 0.0186 | 24 |
| Entail-Base | 0.6128 ± 0.0330 | 0.0771 ± 0.0125 | 24 |
| ROSCOE-LI-Self | 0.7248 ± 0.0364 | 0.0531 ± 0.0104 | 24 |
| ROSCOE-LI-Source* | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 24 |
| ReCEval-Intra | 0.0998 ± 0.0243 | 0.0039 ± 0.0039 | 24 |
| ReCEval-Inter* | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 24 |

\* ROSCOE-LI-Source, ReCEval-Inter: raw scores are exactly 0.0 for every single claim in every ordering (verified: 0/3768 nonzero scores) -- not a code bug, but a real architectural mismatch. These methods score a hypothesis against each individual premise separately and take the minimum (Score = 1 - max(1 - e_i) = min(e_i)); combined with the custom CaptainCookRecipes prompt's explicit instruction to judge a step "Very Unlikely" whenever a required precondition is missing, and every recipe step genuinely requiring *multiple* premises jointly (prior steps + ingredients), almost any single isolated premise will appear to be missing something -- forcing the minimum to 0.0 essentially always. This is inferred from code + prompt inspection, not confirmed against raw model text (only numeric scores are persisted).
