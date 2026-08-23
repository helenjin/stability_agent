# Macro-F1 (paper's convention) vs. Error-Class-Only F1 (Experiment 2's convention)

Same predictions (same CV-thresholded raw scores) scored two ways. Uncertainty is mean +/- bootstrap SD, resampled over graphs (ARES paper's own +/- display style; see analysis/detection_sensitivity.py's bootstrap_sd_over_graphs).

| Method | Paper's Macro-F1 | Ours, Macro-F1 | Ours, Error-Class F1 | Ours, Sound-Class F1 |
|---|---:|---:|---:|---:|
| ARES | 0.633 | 0.583 ± 0.037 | 0.643 ± 0.039 | 0.523 ± 0.051 |
| Entail-Prev | 0.428 | 0.410 ± 0.028 | 0.383 ± 0.027 | 0.437 ± 0.048 |
| Entail-Base | 0.589 | 0.494 ± 0.023 | 0.613 ± 0.033 | 0.376 ± 0.044 |
| ROSCOE-LI-Self | 0.483 | 0.496 ± 0.029 | 0.725 ± 0.036 | 0.266 ± 0.032 |
| ROSCOE-LI-Source | 0.361 | 0.276 ± 0.024 | 0.000 ± 0.000 | 0.552 ± 0.048 |
| ReCEval-Intra | 0.396 | 0.318 ± 0.021 | 0.100 ± 0.024 | 0.537 ± 0.044 |
| ReCEval-Inter | 0.361 | 0.276 ± 0.024 | 0.000 ± 0.000 | 0.552 ± 0.048 |
