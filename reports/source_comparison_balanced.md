# Stability Agent Report

Analyzed 318 runs with low <= 0.3 and high >= 0.8.

- Low-stability runs: 133
- High-stability runs: 85

## Source Summary

| source | n | avg stability | low | high |
| --- | ---: | ---: | ---: | ---: |
| ss_rate_dicts_all | 106 | 0.330 | 60 | 17 |
| ss_rate_dicts_all_nonweb | 106 | 0.318 | 72 | 32 |
| ss_rate_dicts_all_parametric_v0 | 106 | 0.695 | 1 | 36 |

## Source Pair Summary

| left source | right source | n | mean left-right | median left-right | mean abs gap | left higher | right higher | similar | large gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 106 | -0.377 | -0.508 | 0.505 | 21 | 72 | 13 | 80 |
| ss_rate_dicts_all | ss_rate_dicts_all_parametric_v0 | 106 | -0.365 | -0.410 | 0.389 | 9 | 73 | 24 | 57 |
| ss_rate_dicts_all | ss_rate_dicts_all_nonweb | 106 | 0.012 | 0.015 | 0.155 | 38 | 28 | 40 | 24 |

## Radius Summary

| radius | n | avg stability | low | low frac | high | high frac | low lower | low higher | low no dir | low majority share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 318 | 0.393 | 171 | 0.538 | 82 | 0.258 | 101 | 68 | 2 | 0.830 |
| 2 | 318 | 0.412 | 157 | 0.494 | 82 | 0.258 | 94 | 60 | 3 | 0.830 |
| 3 | 318 | 0.426 | 146 | 0.459 | 89 | 0.280 | 91 | 53 | 2 | 0.845 |
| 4 | 318 | 0.439 | 134 | 0.421 | 89 | 0.280 | 86 | 48 | 0 | 0.869 |
| 5 | 318 | 0.449 | 133 | 0.418 | 90 | 0.283 | 86 | 47 | 0 | 0.875 |
| 6 | 315 | 0.455 | 132 | 0.419 | 85 | 0.270 | 85 | 47 | 0 | 0.876 |
| 7 | 314 | 0.458 | 132 | 0.420 | 86 | 0.274 | 85 | 47 | 0 | 0.872 |
| 8 | 311 | 0.458 | 130 | 0.418 | 84 | 0.270 | 83 | 47 | 0 | 0.872 |
| 9 | 305 | 0.467 | 125 | 0.410 | 90 | 0.295 | 80 | 45 | 0 | 0.866 |
| 10 | 301 | 0.473 | 122 | 0.405 | 86 | 0.286 | 77 | 45 | 0 | 0.855 |
| 11 | 288 | 0.476 | 114 | 0.396 | 78 | 0.271 | 73 | 41 | 0 | 0.841 |
| 12 | 279 | 0.484 | 108 | 0.387 | 74 | 0.265 | 67 | 41 | 0 | 0.822 |
| 13 | 264 | 0.477 | 103 | 0.390 | 70 | 0.265 | 63 | 40 | 0 | 0.799 |
| 14 | 245 | 0.472 | 98 | 0.400 | 63 | 0.257 | 59 | 38 | 1 | 0.761 |
| 15 | 225 | 0.478 | 87 | 0.387 | 60 | 0.267 | 55 | 32 | 0 | 0.733 |
| 16 | 193 | 0.481 | 75 | 0.389 | 52 | 0.269 | 47 | 28 | 0 | 0.706 |
| 17 | 164 | 0.485 | 61 | 0.372 | 46 | 0.280 | 41 | 20 | 0 | 0.673 |
| 18 | 139 | 0.458 | 57 | 0.410 | 37 | 0.266 | 37 | 19 | 1 | 0.649 |
| 19 | 112 | 0.457 | 45 | 0.402 | 28 | 0.250 | 32 | 13 | 0 | 0.648 |
| 20 | 82 | 0.415 | 32 | 0.390 | 16 | 0.195 | 21 | 11 | 0 | 0.650 |
| 21 | 61 | 0.414 | 24 | 0.393 | 9 | 0.148 | 17 | 5 | 2 | 0.614 |
| 22 | 41 | 0.363 | 21 | 0.512 | 5 | 0.122 | 16 | 4 | 1 | 0.594 |
| 23 | 25 | 0.368 | 11 | 0.440 | 3 | 0.120 | 10 | 1 | 0 | 0.575 |
| 24 | 19 | 0.295 | 12 | 0.632 | 1 | 0.053 | 10 | 2 | 0 | 0.630 |
| 25 | 10 | 0.340 | 6 | 0.600 | 1 | 0.100 | 6 | 0 | 0 | 0.628 |
| 26 | 6 | 0.352 | 4 | 0.667 | 1 | 0.167 | 3 | 1 | 0 | 0.672 |
| 27 | 5 | 0.433 | 2 | 0.400 | 1 | 0.200 | 1 | 1 | 0 | 0.713 |
| 28 | 4 | 0.378 | 3 | 0.750 | 1 | 0.250 | 2 | 1 | 0 | 0.693 |
| 29 | 4 | 0.368 | 2 | 0.500 | 1 | 0.250 | 1 | 1 | 0 | 0.703 |
| 30 | 4 | 0.373 | 2 | 0.500 | 1 | 0.250 | 1 | 1 | 0 | 0.727 |
| 31 | 3 | 0.460 | 1 | 0.333 | 1 | 0.333 | 1 | 0 | 0 | 1.000 |
| 32 | 3 | 0.471 | 1 | 0.333 | 1 | 0.333 | 1 | 0 | 0 | 1.000 |
| 33 | 3 | 0.473 | 1 | 0.333 | 1 | 0.333 | 1 | 0 | 0 | 1.000 |
| 34 | 3 | 0.462 | 1 | 0.333 | 1 | 0.333 | 1 | 0 | 0 | 1.000 |
| 35 | 3 | 0.496 | 1 | 0.333 | 1 | 0.333 | 1 | 0 | 0 | 1.000 |
| 36 | 3 | 0.453 | 1 | 0.333 | 1 | 0.333 | 1 | 0 | 0 | 1.000 |
| 37 | 3 | 0.451 | 1 | 0.333 | 1 | 0.333 | 1 | 0 | 0 | 1.000 |
| 38 | 1 | 0.420 | 0 | 0.000 | 0 | 0.000 | 0 | 0 | 0 | - |
| 39 | 1 | 0.240 | 1 | 1.000 | 0 | 0.000 | 0 | 0 | 1 | 0.460 |
| 40 | 1 | 0.120 | 1 | 1.000 | 0 | 0.000 | 0 | 0 | 1 | 0.560 |

## Candidate Patterns

| pattern | total | low | high |
| --- | ---: | ---: | ---: |
| added_claims_drive_more_negative_verdict | 65 | 65 | 0 |
| added_claims_drive_more_positive_verdict | 38 | 38 | 0 |
| knife_edge_dependency | 9 | 0 | 4 |
| radius_insensitive_disagreement | 129 | 129 | 0 |

## Redundancy Proxies

| proxy | corr with avg stability | corr with radius-insensitive disagreement |
| --- | ---: | ---: |
| total_claims | -0.319 | 0.417 |
| base_claims | -0.035 | 0.139 |
| addable_claims | -0.254 | 0.275 |
| base_fraction | 0.064 | 0.015 |
| avg_selected | -0.331 | 0.387 |
| avg_selected_fraction | -0.275 | 0.302 |

| group | n | total claims | base claims | addable claims | avg selected fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| radius_insensitive_disagreement | 129 | 20.845 | 3.798 | 17.047 | 0.428 |
| non_radius_insensitive_disagreement | 189 | 17.069 | 2.910 | 14.159 | 0.357 |
| low_stability | 133 | 20.827 | 3.684 | 17.143 | 0.430 |
| high_stability | 85 | 17.941 | 3.847 | 14.094 | 0.362 |

## Semantic Redundancy

Rows with support text: 270

| feature | corr with avg stability | corr with radius-insensitive disagreement |
| --- | ---: | ---: |
| semantic_redundancy_score | -0.011 | -0.005 |
| semantic_independence_score | 0.016 | 0.004 |
| duplicate_pair_fraction | - | - |
| entailment_pair_fraction | - | - |
| same_role_pair_fraction | -0.005 | -0.018 |
| complementary_pair_fraction | 0.024 | 0.013 |
| tension_pair_fraction | -0.024 | -0.007 |
| redundant_sentence_fraction | -0.016 | -0.004 |
| support_concentration | -0.015 | 0.010 |
| effective_support_count | -0.034 | 0.038 |
| independent_support_count | -0.040 | 0.043 |
| redundancy_excess | -0.026 | 0.009 |

| group | n | text count | semantic redundancy | independence | same-role pairs | duplicate pairs | tension pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| radius_insensitive_disagreement | 111 | 4.991 | 0.023 | 0.986 | 0.006 | 0.000 | 0.044 |
| non_radius_insensitive_disagreement | 159 | 4.912 | 0.023 | 0.986 | 0.007 | 0.000 | 0.045 |
| low_stability | 114 | 4.991 | 0.023 | 0.987 | 0.006 | 0.000 | 0.043 |
| high_stability | 69 | 4.913 | 0.022 | 0.992 | 0.004 | 0.000 | 0.035 |

## Lowest Stability Examples

| example | source | avg | first | last | majority | direction | patterns |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 13 | ss_rate_dicts_all | 0.000 | 0.000 | 0.000 | 2.0 (0.954) | higher | added_claims_drive_more_positive_verdict, radius_insensitive_disagreement |
| 43 | ss_rate_dicts_all | 0.000 | 0.000 | 0.000 | 2.0 (0.951) | higher | added_claims_drive_more_positive_verdict, radius_insensitive_disagreement |
| 52 | ss_rate_dicts_all | 0.000 | 0.000 | 0.000 | 1.0 (0.864) | higher | added_claims_drive_more_positive_verdict, radius_insensitive_disagreement |
| 54 | ss_rate_dicts_all | 0.000 | 0.000 | 0.000 | 1.0 (0.584) | higher | radius_insensitive_disagreement |
| 59 | ss_rate_dicts_all | 0.000 | 0.000 | 0.000 | 1.0 (0.537) | higher | radius_insensitive_disagreement |
| 66 | ss_rate_dicts_all | 0.000 | 0.000 | 0.000 | 2.0 (0.958) | higher | added_claims_drive_more_positive_verdict, radius_insensitive_disagreement |
| 74 | ss_rate_dicts_all | 0.000 | 0.000 | 0.000 | 0.0 (0.849) | higher | added_claims_drive_more_positive_verdict, radius_insensitive_disagreement |
| 0 | ss_rate_dicts_all_nonweb | 0.000 | 0.000 | 0.000 | -2.0 (1.000) | lower | added_claims_drive_more_negative_verdict, radius_insensitive_disagreement |
| 2 | ss_rate_dicts_all_nonweb | 0.000 | 0.000 | 0.000 | -1.0 (1.000) | lower | added_claims_drive_more_negative_verdict, radius_insensitive_disagreement |
| 3 | ss_rate_dicts_all_nonweb | 0.000 | 0.000 | 0.000 | 1.0 (1.000) | lower | added_claims_drive_more_negative_verdict, radius_insensitive_disagreement |
| 5 | ss_rate_dicts_all_nonweb | 0.000 | 0.000 | 0.000 | -1.0 (1.000) | higher | added_claims_drive_more_positive_verdict, radius_insensitive_disagreement |
| 6 | ss_rate_dicts_all_nonweb | 0.000 | 0.000 | 0.000 | -1.0 (1.000) | lower | added_claims_drive_more_negative_verdict, radius_insensitive_disagreement |

## Highest Stability Examples

| example | source | avg | first | last | majority | direction | patterns |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | 2.0 (1.000) | - | - |
| 4 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -2.0 (1.000) | - | - |
| 10 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -1.0 (1.000) | - | - |
| 14 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -1.0 (1.000) | - | - |
| 16 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -2.0 (1.000) | - | - |
| 21 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -2.0 (1.000) | - | - |
| 23 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -2.0 (1.000) | - | - |
| 27 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | 2.0 (1.000) | - | - |
| 28 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -2.0 (1.000) | - | - |
| 29 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -2.0 (1.000) | - | - |
| 31 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -1.0 (1.000) | - | - |
| 35 | ss_rate_dicts_all_nonweb | 1.000 | 1.000 | 1.000 | -2.0 (1.000) | - | - |

## Cross-Source Sensitivity

| example | least stable | most stable | gap | pattern |
| --- | --- | --- | ---: | --- |
| 13 | ss_rate_dicts_all | ss_rate_dicts_all_parametric_v0 | 1.000 | ss_rate_dicts_all_sensitive_instability |
| 43 | ss_rate_dicts_all | ss_rate_dicts_all_parametric_v0 | 1.000 | ss_rate_dicts_all_sensitive_instability |
| 49 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 1.000 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 66 | ss_rate_dicts_all | ss_rate_dicts_all_parametric_v0 | 1.000 | ss_rate_dicts_all_sensitive_instability |
| 7 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 0.996 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 57 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 0.975 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 93 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 0.975 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 73 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 0.964 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 85 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 0.930 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 34 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 0.921 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 99 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | 0.918 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 52 | ss_rate_dicts_all | ss_rate_dicts_all_parametric_v0 | 0.911 | ss_rate_dicts_all_sensitive_instability |

## Reading

Pattern tags are hypotheses for triage. They indicate where low-stability examples differ from high-stability examples in the sampled verdict distribution; they do not by themselves prove the underlying reasoning error.
