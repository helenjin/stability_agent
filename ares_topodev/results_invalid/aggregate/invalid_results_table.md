# Experiment 2: Dependency-Violation Sensitivity

`v` = node whose true prerequisite got displaced (claim text still names it, but it's no longer in the premise prefix). `u` = the displaced prerequisite itself.

`frac_exceeds_noise` = fraction of cases where the violation moved the score by more than Experiment 1's already-measured valid-reordering range for that node -- i.e. the violation's effect stands out above ordinary harmless-reordering noise, rather than being lost in it.

`frac_correct_direction` = fraction of cases where the score actually dropped (not just changed) when the dependency was violated -- the expected direction for a method that's tracking validity.

| Method | N cases | Exceeds noise (v) | Correct direction (v) | Median \|effect\|/TopoDev (v) | Exceeds noise (u) | Correct direction (u) | Median \|effect\|/TopoDev (u) |
|---|---|---|---|---|---|---|---|
| ARES | 245 | 0.065 | 0.347 | 0.156 | 0.110 | 0.327 | 0.179 |
| Entail-Prev | 245 | 0.404 | 0.522 | 0.000 | 0.229 | 0.310 | 0.000 |
| Entail-Base | 245 | 0.082 | 0.082 | 1.000 | 0.082 | 0.065 | 0.000 |
