# Main Results Table

TopoDev = mean max-min range per step across sampled orderings (sensitive to a single outlier ordering).
TopoVar = mean per-step sample variance across sampled orderings (more robust to a single outlier, but on a squared-score scale).

| Dataset | Method | TopoDev (mean) | TopoDev 95% CI | TopoVar (mean) | TopoVar 95% CI | N |
|---|---|---|---|---|---|---|
| ClaimTrees | ARES | N/A | N/A | N/A | N/A | 0 | -- every released ClaimTrees config is a strict linear chain (exactly one valid topological order); TopoDev/TopoVar are structurally undefined, not merely unmeasured. See VENDOR.md.
| ClaimTrees | Entail-Prev | N/A | N/A | N/A | N/A | 0 | -- every released ClaimTrees config is a strict linear chain (exactly one valid topological order); TopoDev/TopoVar are structurally undefined, not merely unmeasured. See VENDOR.md.
| ClaimTrees | Entail-Base | N/A | N/A | N/A | N/A | 0 | -- every released ClaimTrees config is a strict linear chain (exactly one valid topological order); TopoDev/TopoVar are structurally undefined, not merely unmeasured. See VENDOR.md.
| CaptainCookRecipes | ARES | 0.3334 | [0.2777, 0.3889] | 0.0243 | [0.0186, 0.0304] | 24 |
| CaptainCookRecipes | Entail-Prev | 0.4069 | [0.3435, 0.4656] | 0.0809 | [0.0678, 0.0931] | 24 |
| CaptainCookRecipes | Entail-Base | 0.1483 | [0.1081, 0.1923] | 0.0319 | [0.0234, 0.0409] | 24 |

## Error-Recovery Consistency

Does a method flag the *same* ground-truth error node(s) as errors regardless of which valid ordering presented them? Threshold per method chosen to maximize pooled Macro-F1 against ground truth (see analysis/error_recovery.py) -- a simplification of the paper's cross-validated procedure, not a reproduction of it. `jaccard`=1.0 means the predicted-error node set never changes across orderings for a recipe; `exact_match_gt`=1.0 means every ordering's predicted set exactly equals the true error set.

| Method | Threshold | Mean pairwise Jaccard (predicted-set stability) | Frac. orderings exactly matching ground truth | Mean recall | Mean precision | N |
|---|---|---|---|---|---|---|
| ARES | 0.1745 | 0.7340 | 0.0042 | 0.6837 | 0.6722 | 24 |
| Entail-Prev | 0.4000 | 0.6041 | 0.0000 | 0.3356 | 0.5455 | 24 |
| Entail-Base | 1.0000 | 0.9026 | 0.0000 | 0.6896 | 0.6131 | 24 |
