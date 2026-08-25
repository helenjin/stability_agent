# Experiment 2 Main Table: Error-Detection Sensitivity Under Valid Reordering

Uncertainty is mean +/- bootstrap SD, resampled over graphs (10000 resamples, ARES paper's own +/- display style; see analysis/detection_sensitivity.py's bootstrap_sd_over_graphs).

| Method | Mean F1 ↑ | ΔF1 ↓ | N graphs |
|---|---:|---:|---:|
| ARES | 0.6292 ± 0.0423 | 0.2656 ± 0.0392 | 24 |
| sager | 0.5688 ± 0.0435 | 0.1650 ± 0.0236 | 24 |
