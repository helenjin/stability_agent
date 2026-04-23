# Stability Agent Report

Analyzed 246 runs with low <= 0.3 and high >= 0.8.

- Low-stability runs: 132
- High-stability runs: 61

## Source Summary

| source | n | avg stability | low | high |
| --- | ---: | ---: | ---: | ---: |
| ss_rate_dicts_all | 106 | 0.330 | 60 | 17 |
| ss_rate_dicts_all_nonweb | 106 | 0.318 | 72 | 32 |
| ss_rate_dicts_all_parametric | 34 | 0.682 | 0 | 12 |

## Radius Summary

| radius | n | avg stability | low | low frac | high | high frac | low lower | low higher | low no dir | low majority share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 246 | 0.350 | 146 | 0.593 | 62 | 0.252 | 91 | 54 | 1 | 0.865 |
| 2 | 246 | 0.359 | 142 | 0.577 | 62 | 0.252 | 90 | 49 | 3 | 0.865 |
| 3 | 246 | 0.365 | 139 | 0.565 | 62 | 0.252 | 89 | 49 | 1 | 0.868 |
| 4 | 246 | 0.368 | 138 | 0.561 | 64 | 0.260 | 89 | 49 | 0 | 0.868 |
| 5 | 246 | 0.375 | 135 | 0.549 | 63 | 0.256 | 87 | 48 | 0 | 0.872 |
| 6 | 243 | 0.373 | 133 | 0.547 | 63 | 0.259 | 86 | 47 | 0 | 0.877 |
| 7 | 242 | 0.374 | 133 | 0.550 | 61 | 0.252 | 87 | 46 | 0 | 0.872 |
| 8 | 239 | 0.373 | 129 | 0.540 | 60 | 0.251 | 83 | 46 | 0 | 0.875 |
| 9 | 233 | 0.382 | 124 | 0.532 | 61 | 0.262 | 80 | 44 | 0 | 0.868 |
| 10 | 229 | 0.387 | 121 | 0.528 | 60 | 0.262 | 77 | 44 | 0 | 0.858 |
| 11 | 215 | 0.385 | 113 | 0.526 | 53 | 0.247 | 73 | 40 | 0 | 0.844 |
| 12 | 208 | 0.393 | 107 | 0.514 | 48 | 0.231 | 67 | 40 | 0 | 0.824 |
| 13 | 201 | 0.391 | 102 | 0.507 | 47 | 0.234 | 63 | 39 | 0 | 0.803 |
| 14 | 184 | 0.379 | 97 | 0.527 | 41 | 0.223 | 59 | 37 | 1 | 0.764 |
| 15 | 164 | 0.382 | 86 | 0.524 | 36 | 0.220 | 55 | 31 | 0 | 0.736 |
| 16 | 142 | 0.382 | 74 | 0.521 | 30 | 0.211 | 47 | 27 | 0 | 0.708 |
| 17 | 125 | 0.388 | 60 | 0.480 | 27 | 0.216 | 41 | 19 | 0 | 0.675 |
| 18 | 111 | 0.374 | 56 | 0.505 | 21 | 0.189 | 37 | 18 | 1 | 0.651 |
| 19 | 93 | 0.380 | 45 | 0.484 | 15 | 0.161 | 32 | 13 | 0 | 0.648 |
| 20 | 72 | 0.377 | 32 | 0.444 | 14 | 0.194 | 21 | 11 | 0 | 0.650 |
| 21 | 52 | 0.368 | 24 | 0.462 | 7 | 0.135 | 17 | 5 | 2 | 0.614 |
| 22 | 38 | 0.335 | 21 | 0.553 | 5 | 0.132 | 16 | 4 | 1 | 0.594 |
| 23 | 26 | 0.386 | 11 | 0.423 | 3 | 0.115 | 10 | 1 | 0 | 0.575 |
| 24 | 20 | 0.306 | 12 | 0.600 | 1 | 0.050 | 10 | 2 | 0 | 0.630 |
| 25 | 9 | 0.287 | 6 | 0.667 | 0 | 0.000 | 6 | 0 | 0 | 0.628 |
| 26 | 5 | 0.255 | 4 | 0.800 | 0 | 0.000 | 3 | 1 | 0 | 0.672 |
| 27 | 4 | 0.347 | 2 | 0.500 | 1 | 0.250 | 1 | 1 | 0 | 0.713 |
| 28 | 3 | 0.171 | 3 | 1.000 | 0 | 0.000 | 2 | 1 | 0 | 0.693 |
| 29 | 3 | 0.162 | 2 | 0.667 | 0 | 0.000 | 1 | 1 | 0 | 0.703 |
| 30 | 3 | 0.169 | 2 | 0.667 | 0 | 0.000 | 1 | 1 | 0 | 0.727 |
| 31 | 2 | 0.193 | 1 | 0.500 | 0 | 0.000 | 1 | 0 | 0 | 1.000 |
| 32 | 2 | 0.213 | 1 | 0.500 | 0 | 0.000 | 1 | 0 | 0 | 1.000 |
| 33 | 2 | 0.213 | 1 | 0.500 | 0 | 0.000 | 1 | 0 | 0 | 1.000 |
| 34 | 2 | 0.200 | 1 | 0.500 | 0 | 0.000 | 1 | 0 | 0 | 1.000 |
| 35 | 2 | 0.247 | 1 | 0.500 | 0 | 0.000 | 1 | 0 | 0 | 1.000 |
| 36 | 2 | 0.190 | 1 | 0.500 | 0 | 0.000 | 1 | 0 | 0 | 1.000 |
| 37 | 2 | 0.183 | 1 | 0.500 | 0 | 0.000 | 1 | 0 | 0 | 1.000 |
| 38 | 1 | 0.420 | 0 | 0.000 | 0 | 0.000 | 0 | 0 | 0 | - |
| 39 | 1 | 0.240 | 1 | 1.000 | 0 | 0.000 | 0 | 0 | 1 | 0.460 |
| 40 | 1 | 0.120 | 1 | 1.000 | 0 | 0.000 | 0 | 0 | 1 | 0.560 |

## Candidate Patterns

| pattern | total | low | high |
| --- | ---: | ---: | ---: |
| added_claims_drive_more_negative_verdict | 65 | 65 | 0 |
| added_claims_drive_more_positive_verdict | 38 | 38 | 0 |
| knife_edge_dependency | 9 | 0 | 4 |
| radius_insensitive_disagreement | 128 | 128 | 0 |

## Redundancy Proxies

| proxy | corr with avg stability | corr with radius-insensitive disagreement |
| --- | ---: | ---: |
| total_claims | -0.029 | 0.032 |
| base_claims | -0.028 | 0.159 |
| addable_claims | -0.001 | -0.094 |
| base_fraction | -0.014 | 0.150 |
| avg_selected | -0.167 | 0.206 |
| avg_selected_fraction | -0.162 | 0.206 |

| group | n | total claims | base claims | addable claims | avg selected fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| radius_insensitive_disagreement | 128 | 20.898 | 3.805 | 17.094 | 0.429 |
| non_radius_insensitive_disagreement | 118 | 20.661 | 2.636 | 18.025 | 0.370 |
| low_stability | 132 | 20.879 | 3.689 | 17.189 | 0.431 |
| high_stability | 61 | 20.426 | 4.246 | 16.180 | 0.386 |

## Semantic Redundancy

Rows with support text: 212

| feature | corr with avg stability | corr with radius-insensitive disagreement |
| --- | ---: | ---: |
| semantic_redundancy_score | -0.010 | -0.012 |
| semantic_independence_score | 0.007 | 0.025 |
| duplicate_pair_fraction | - | - |
| entailment_pair_fraction | - | - |
| same_role_pair_fraction | 0.006 | -0.042 |
| complementary_pair_fraction | 0.004 | 0.027 |
| tension_pair_fraction | -0.007 | -0.013 |
| redundant_sentence_fraction | -0.007 | -0.025 |
| support_concentration | -0.026 | 0.023 |
| effective_support_count | -0.026 | 0.035 |
| independent_support_count | -0.032 | 0.040 |
| redundancy_excess | -0.018 | -0.010 |

| group | n | text count | semantic redundancy | independence | same-role pairs | duplicate pairs | tension pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| radius_insensitive_disagreement | 110 | 4.991 | 0.023 | 0.986 | 0.006 | 0.000 | 0.043 |
| non_radius_insensitive_disagreement | 102 | 4.931 | 0.024 | 0.983 | 0.008 | 0.000 | 0.045 |
| low_stability | 113 | 4.991 | 0.023 | 0.986 | 0.006 | 0.000 | 0.042 |
| high_stability | 50 | 4.920 | 0.023 | 0.989 | 0.005 | 0.000 | 0.037 |

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
| 9 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 1.000 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 34 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 1.000 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 13 | ss_rate_dicts_all | ss_rate_dicts_all_parametric | 0.954 | ss_rate_dicts_all_sensitive_instability |
| 85 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 0.916 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 36 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 0.803 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 0 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 0.773 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 7 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 0.701 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 69 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 0.697 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 12 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 0.672 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 31 | ss_rate_dicts_all_parametric | ss_rate_dicts_all_nonweb | 0.645 | ss_rate_dicts_all_parametric_sensitive_instability |
| 15 | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric | 0.639 | ss_rate_dicts_all_nonweb_sensitive_instability |
| 47 | ss_rate_dicts_all | ss_rate_dicts_all_nonweb | 0.566 | ss_rate_dicts_all_sensitive_instability |

## Reading

Pattern tags are hypotheses for triage. They indicate where low-stability examples differ from high-stability examples in the sampled verdict distribution; they do not by themselves prove the underlying reasoning error.
