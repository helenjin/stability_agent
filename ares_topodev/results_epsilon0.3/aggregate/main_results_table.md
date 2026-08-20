# Main Results Table

TopoDev = mean max-min range per step across sampled orderings (sensitive to a single outlier ordering).
TopoVar = mean per-step sample variance across sampled orderings (more robust to a single outlier, but on a squared-score scale).

| Dataset | Method | TopoDev (mean) | TopoDev 95% CI | TopoVar (mean) | TopoVar 95% CI | N |
|---|---|---|---|---|---|---|
| ClaimTrees | ARES | N/A | N/A | N/A | N/A | 0 | -- every released ClaimTrees config is a strict linear chain (exactly one valid topological order); TopoDev/TopoVar are structurally undefined, not merely unmeasured. See VENDOR.md.
| ClaimTrees | Entail-Prev | N/A | N/A | N/A | N/A | 0 | -- every released ClaimTrees config is a strict linear chain (exactly one valid topological order); TopoDev/TopoVar are structurally undefined, not merely unmeasured. See VENDOR.md.
| ClaimTrees | Entail-Base | N/A | N/A | N/A | N/A | 0 | -- every released ClaimTrees config is a strict linear chain (exactly one valid topological order); TopoDev/TopoVar are structurally undefined, not merely unmeasured. See VENDOR.md.
| CaptainCookRecipes | ARES | 0.4427 | [0.3863, 0.4964] | 0.0402 | [0.0332, 0.0471] | 24 |
| CaptainCookRecipes | Entail-Prev | 0.4164 | [0.3532, 0.4797] | 0.0816 | [0.0686, 0.0947] | 24 |
| CaptainCookRecipes | Entail-Base | 0.1345 | [0.1016, 0.1691] | 0.0281 | [0.0205, 0.0370] | 24 |

## Error-Recovery Consistency

Does a method flag the *same* ground-truth error node(s) as errors regardless of which valid ordering presented them? Threshold per method chosen to maximize pooled Macro-F1 against ground truth (see analysis/error_recovery.py) -- a simplification of the paper's cross-validated procedure, not a reproduction of it. `jaccard`=1.0 means the predicted-error node set never changes across orderings for a recipe; `exact_match_gt`=1.0 means every ordering's predicted set exactly equals the true error set.

| Method | Threshold | Mean pairwise Jaccard (predicted-set stability) | Frac. orderings exactly matching ground truth | Mean recall | Mean precision | N |
|---|---|---|---|---|---|---|
| ARES | 0.0312 | 0.6170 | 0.0083 | 0.6022 | 0.6998 | 24 |
| Entail-Prev | 0.4000 | 0.6134 | 0.0000 | 0.3362 | 0.5423 | 24 |
| Entail-Base | 1.0000 | 0.9108 | 0.0000 | 0.6695 | 0.6119 | 24 |
