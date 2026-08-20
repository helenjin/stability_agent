# Experiment 2: Dependency-Violation Sensitivity

`v` = node whose true prerequisite got displaced (claim text still names it, but it's no longer in the premise prefix). `u` = the displaced prerequisite itself.

`frac_exceeds_noise` = fraction of cases where the violation moved the score by more than Experiment 1's already-measured valid-reordering range for that node -- i.e. the violation's effect stands out above ordinary harmless-reordering noise, rather than being lost in it.

`frac_correct_direction` = fraction of cases where the score actually dropped (not just changed) when the dependency was violated -- the expected direction for a method that's tracking validity.

| Method | N cases | Exceeds noise (v) | Correct direction (v) | Median \|effect\|/TopoDev (v) | Exceeds noise (u) | Correct direction (u) | Median \|effect\|/TopoDev (u) |
|---|---|---|---|---|---|---|---|
| ARES | 245 | 0.151 | 0.588 | 0.230 | 0.163 | 0.494 | 0.261 |
| Entail-Prev | 245 | 0.388 | 0.531 | 0.500 | 0.237 | 0.318 | 0.000 |
| Entail-Base | 245 | 0.082 | 0.061 | 1.000 | 0.090 | 0.053 | 0.000 |
