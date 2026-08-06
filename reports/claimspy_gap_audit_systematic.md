# ClaimSpy Large-Gap Audit (Systematic)

Of 106 matched examples, **83 have a cross-source stability gap >= 0.30** (gap = max - min avg_stability across sources).

## Headline

- `ss_rate_dicts_all_parametric_v0` is the most stable source in 65 of 83 gap cases (mean stability 0.651; >= 0.80 in 18, 0.50-0.80 in 46, < 0.50 in 19).
- `ss_rate_dicts_all_nonweb` is the most stable source in 12 gap cases.
- `ss_rate_dicts_all` is the most stable source in 6 gap cases.
- Disagreement direction on the least-stable side: `higher` in 33, `lower` in 48.

## Signatures (data patterns, not confirmed causes)

| signature | n |
| --- | ---: |
| retrieval_oversensitivity | 64 |
| retrieved_more_stable | 18 |
| parametric_more_stable | 1 |

> Caveat: these are signatures in the serialized outputs, not verified mechanisms.

## By Domain

| domain | gap cases | mean gap |
| --- | ---: | ---: |
| gtri | 16 | 0.683 |
| unknown | 14 | 0.631 |
| str_speculative | 6 | 0.589 |
| umbc_alloys | 5 | 0.581 |
| umbc_batteries | 4 | 0.714 |
| umbc_semiconductors | 4 | 0.745 |
| matsci_db | 4 | 0.533 |
| umbc_superconductors | 4 | 0.648 |
| str_dft | 4 | 0.552 |
| mof_alloys | 3 | 0.823 |
| mof_superconductors | 3 | 0.728 |
| vis_reas_alloys | 3 | 0.576 |
| vis_reas_batteries | 3 | 0.472 |
| str_pvskt | 2 | 0.508 |
| mof_batteries | 2 | 0.580 |
| vis_reas_superconductors | 2 | 0.534 |
| mof_semiconductors | 2 | 0.500 |
| str_spectulative | 1 | 0.975 |
| vis_reas_semiconductors | 1 | 0.669 |

## Top 12 Largest Gaps

| ex | problem | domain | q | ss_rate_dicts_all | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | gap | dir | cell |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 13 | mof_alloys_0008 | mof_alloys | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | higher | stable-correct |
| 43 | umbc_batteries_0003 | umbc_batteries | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | higher | stable-correct |
| 49 | umbc_batteries_0004 | umbc_batteries | 0.000 | 0.001 | 0.000 | 1.000 | 1.000 | lower | stable-wrong |
| 66 | gtri_0001 | gtri | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | higher | stable-correct |
| 7 | mof_superconductors_0002 | mof_superconductors | 0.000 | 0.119 | 0.004 | 1.000 | 0.996 | lower | stable-wrong |
| 57 | jhu-alloys-0008 | unknown | 0.000 | 0.010 | 0.000 | 0.975 | 0.975 | lower | stable-wrong |
| 93 | str_spectulative_01 | str_spectulative | 0.000 | 0.215 | 0.000 | 0.975 | 0.975 | lower | stable-wrong |
| 73 | gtri_0008 | gtri | 1.000 | 0.001 | 0.000 | 0.964 | 0.964 | higher | stable-correct |
| 85 | gtri_0020 | gtri | 0.500 | 0.043 | 0.000 | 0.930 | 0.930 | lower | stable-mid |
| 34 | umbc_semiconductors_0001 | umbc_semiconductors | 0.000 | 0.030 | 0.000 | 0.921 | 0.921 | lower | stable-wrong |
| 99 | str_speculative_11 | str_speculative | 0.000 | 0.005 | 0.000 | 0.918 | 0.918 | lower | stable-wrong |
| 52 | jhu-alloys-0003 | unknown | 0.750 | 0.000 | 0.000 | 0.911 | 0.911 | higher | stable-correct |

## Stable-Wrong Cases (parametric stable, quality low)

Parametric reasoning that is perturbation-robust **and** low-quality -- rigidity, not reliability.

| ex | problem | domain | param | quality | dir |
| ---: | --- | --- | ---: | ---: | --- |
| 49 | umbc_batteries_0004 | umbc_batteries | 1.000 | 0.000 | lower |
| 7 | mof_superconductors_0002 | mof_superconductors | 1.000 | 0.000 | lower |
| 57 | jhu-alloys-0008 | unknown | 0.975 | 0.000 | lower |
| 93 | str_spectulative_01 | str_spectulative | 0.975 | 0.000 | lower |
| 34 | umbc_semiconductors_0001 | umbc_semiconductors | 0.921 | 0.000 | lower |
| 99 | str_speculative_11 | str_speculative | 0.918 | 0.000 | lower |
| 30 | umbc_alloys_0001 | umbc_alloys | 0.855 | 0.000 | lower |
| 77 | gtri_0012 | gtri | 0.851 | 0.000 | lower |

## Counter-Examples: Retrieval More Stable (18)

| ex | problem | domain | q | ss_rate_dicts_all | ss_rate_dicts_all_nonweb | ss_rate_dicts_all_parametric_v0 | best |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 47 | umbc_superconductors_0004 | umbc_superconductors | 0.250 | 0.434 | 1.000 | 0.380 | ss_rate_dicts_all_nonweb |
| 10 | mof_superconductors_0005 | mof_superconductors | 0.250 | 0.532 | 1.000 | 0.406 | ss_rate_dicts_all_nonweb |
| 88 | str_dft_04 | str_dft | 0.250 | 0.498 | 1.000 | 0.464 | ss_rate_dicts_all_nonweb |
| 58 | jhu-alloys-0009 | unknown | 0.750 | 0.521 | 0.002 | 0.454 | ss_rate_dicts_all |
| 62 | jhu-alloys-0013 | unknown | 0.750 | 0.669 | 1.000 | 0.487 | ss_rate_dicts_all_nonweb |
| 8 | mof_semiconductors_0003 | mof_semiconductors | 0.750 | 0.510 | 0.000 | 0.472 | ss_rate_dicts_all |
| 14 | mof_semiconductors_0009 | mof_semiconductors | 0.250 | 0.569 | 1.000 | 0.509 | ss_rate_dicts_all_nonweb |
| 3 | matsci_db_004 | matsci_db | 0.750 | 0.489 | 0.000 | 0.448 | ss_rate_dicts_all |
| 20 | vis_reas_batteries_0001 | vis_reas_batteries | 0.750 | 0.462 | 0.000 | 0.413 | ss_rate_dicts_all |
| 71 | gtri_0006 | gtri | 0.000 | 0.544 | 1.000 | 0.634 | ss_rate_dicts_all_nonweb |
| 31 | umbc_alloys_0002 | umbc_alloys | 0.250 | 0.574 | 1.000 | 0.550 | ss_rate_dicts_all_nonweb |
| 55 | jhu-alloys-0006 | unknown | 0.750 | 0.665 | 1.000 | 0.572 | ss_rate_dicts_all_nonweb |
| 23 | vis_reas_batteries_0004 | vis_reas_batteries | 0.000 | 0.592 | 1.000 | 0.698 | ss_rate_dicts_all_nonweb |
| 61 | jhu-alloys-0012 | unknown | 0.750 | 0.598 | 1.000 | 0.639 | ss_rate_dicts_all_nonweb |
| 64 | jhu-superconductors-0002 | unknown | 0.250 | 0.550 | 0.179 | 0.383 | ss_rate_dicts_all |
| 90 | str_dft_06 | str_dft | 0.250 | 0.652 | 1.000 | 0.757 | ss_rate_dicts_all_nonweb |
| 81 | gtri_0016 | gtri | 0.000 | 0.653 | 1.000 | 0.681 | ss_rate_dicts_all_nonweb |
| 33 | umbc_alloys_0004 | umbc_alloys | 0.500 | 0.430 | 0.127 | 0.362 | ss_rate_dicts_all |

## Limitations

- Quality label is a proxy. ClaimSpy `continuous_score` is treated as ground-truth correctness; it is itself a model-assisted judgment.
- Signatures are not causes. The error-mode taxonomy (stale memory, conflicting/noisy retrieval, sparse support, multi-hop brittleness) cannot be confirmed from serialized labels alone.
