# ClaimSpy Source Follow-up

## Source Summary

| source | avg stability | low | high |
| --- | ---: | ---: | ---: |
| ss_rate_dicts_all | 0.330 | 60 | 17 |
| ss_rate_dicts_all_nonweb | 0.318 | 72 | 32 |
| ss_rate_dicts_all_parametric_v0 | 0.695 | 1 | 36 |

## Source Pair Summary

| left source | right source | mean left-right | mean abs gap | left higher | right higher | similar |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | -0.377 | 0.505 | 21 | 72 | 13 |
| ss_rate_dicts_all | ss_rate_dicts_all_parametric_v0 | -0.365 | 0.389 | 9 | 73 | 24 |
| ss_rate_dicts_all | ss_rate_dicts_all_nonweb | 0.012 | 0.155 | 38 | 28 | 40 |

## Domain by Source

| domain | source | n | avg stability | avg quality | low | high |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| alloys | ss_rate_dicts_all | 16 | 0.307 | 0.250 | 9 | 2 |
| alloys | ss_rate_dicts_all_nonweb | 16 | 0.250 | 0.250 | 12 | 4 |
| alloys | ss_rate_dicts_all_parametric_v0 | 16 | 0.651 | 0.250 | 0 | 4 |
| alloys_sup | ss_rate_dicts_all | 4 | 0.406 | 0.250 | 1 | 0 |
| alloys_sup | ss_rate_dicts_all_nonweb | 4 | 0.269 | 0.250 | 3 | 1 |
| alloys_sup | ss_rate_dicts_all_parametric_v0 | 4 | 0.689 | 0.250 | 0 | 1 |
| batteries | ss_rate_dicts_all | 5 | 0.416 | 0.250 | 2 | 1 |
| batteries | ss_rate_dicts_all_nonweb | 5 | 0.400 | 0.250 | 3 | 2 |
| batteries | ss_rate_dicts_all_parametric_v0 | 5 | 0.661 | 0.250 | 0 | 1 |
| computational_tools | ss_rate_dicts_all | 37 | 0.327 | 0.419 | 22 | 7 |
| computational_tools | ss_rate_dicts_all_nonweb | 37 | 0.333 | 0.419 | 25 | 12 |
| computational_tools | ss_rate_dicts_all_parametric_v0 | 37 | 0.728 | 0.419 | 0 | 15 |
| modalities | ss_rate_dicts_all | 8 | 0.300 | 0.376 | 5 | 1 |
| modalities | ss_rate_dicts_all_nonweb | 8 | 0.276 | 0.376 | 6 | 2 |
| modalities | ss_rate_dicts_all_parametric_v0 | 8 | 0.663 | 0.376 | 0 | 3 |
| semiconductors | ss_rate_dicts_all | 5 | 0.201 | 0.000 | 3 | 0 |
| semiconductors | ss_rate_dicts_all_nonweb | 5 | 0.200 | 0.000 | 4 | 1 |
| semiconductors | ss_rate_dicts_all_parametric_v0 | 5 | 0.651 | 0.000 | 0 | 1 |
| superconductors | ss_rate_dicts_all | 15 | 0.289 | 0.300 | 9 | 2 |
| superconductors | ss_rate_dicts_all_nonweb | 15 | 0.339 | 0.300 | 9 | 4 |
| superconductors | ss_rate_dicts_all_parametric_v0 | 15 | 0.678 | 0.300 | 1 | 5 |
| unknown | ss_rate_dicts_all | 16 | 0.406 | - | 9 | 4 |
| unknown | ss_rate_dicts_all_nonweb | 16 | 0.376 | - | 10 | 6 |
| unknown | ss_rate_dicts_all_parametric_v0 | 16 | 0.721 | - | 0 | 6 |

## Stability vs Quality

| source | bucket | n | avg stability | avg quality |
| --- | --- | ---: | ---: | ---: |
| ss_rate_dicts_all | high_quality | 30 | 0.330 | 0.958 |
| ss_rate_dicts_all | low_quality | 60 | 0.309 | 0.009 |
| ss_rate_dicts_all | missing | 16 | 0.406 | - |
| ss_rate_dicts_all_nonweb | high_quality | 30 | 0.344 | 0.958 |
| ss_rate_dicts_all_nonweb | low_quality | 60 | 0.290 | 0.009 |
| ss_rate_dicts_all_nonweb | missing | 16 | 0.376 | - |
| ss_rate_dicts_all_parametric_v0 | high_quality | 30 | 0.695 | 0.958 |
| ss_rate_dicts_all_parametric_v0 | low_quality | 60 | 0.688 | 0.009 |
| ss_rate_dicts_all_parametric_v0 | missing | 16 | 0.721 | - |

## Largest Source Gaps

| example | problem | domain | most stable | least stable | gap | quality |
| --- | --- | --- | --- | --- | ---: | ---: |
| 13 | alloys_0014 | alloys | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all | 1.000 | 1.000 |
| 43 | computational_tools_0019 | computational_tools | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all | 1.000 | 0.000 |
| 49 | computational_tools_0025 | computational_tools | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all_nonweb | 1.000 | 0.000 |
| 66 | modalities_0003 | modalities | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all | 1.000 | 1.000 |
| 7 | alloys_0008 | alloys | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all_nonweb | 0.996 | 0.000 |
| 57 | computational_tools_0033 | computational_tools | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all_nonweb | 0.975 | 0.250 |
| 93 | - | - | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all_nonweb | 0.975 | - |
| 73 | semiconductors_0004 | semiconductors | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all_nonweb | 0.964 | 0.000 |
| 85 | superconductors_0011 | superconductors | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all_nonweb | 0.930 | 0.000 |
| 34 | computational_tools_0010 | computational_tools | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all_nonweb | 0.921 | 1.000 |
| 99 | - | - | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all_nonweb | 0.918 | - |
| 52 | computational_tools_0028 | computational_tools | ss_rate_dicts_all_parametric_v0 | ss_rate_dicts_all | 0.911 | 1.000 |

## Reading

These summaries are diagnostic. A higher stability score means the judgment changes less under evidence perturbation; it does not automatically mean the judgment is more correct.
