# Error-detection breakdown by causal relationship to the injected error

Source/propagated rows: only recall is meaningful (these categories are all ground-truth errors -- precision is trivially 1.0). Ancestor/independent rows: only false-positive rate is meaningful (these categories are all ground-truth sound -- precision is trivially 0.0).

| Method | Category | n | Recall | FP rate |
|---|---|---:|---:|---:|
| ares | source | 405 | 0.7037 | -- |
| ares | propagated | 1841 | 0.6757 | -- |
| ares | ancestor_of_source | 910 | -- | 0.3890 |
| ares | independent | 612 | -- | 0.4771 |
| sager | source | 405 | 0.5951 | -- |
| sager | propagated | 1841 | 0.6089 | -- |
| sager | ancestor_of_source | 910 | -- | 0.4088 |
| sager | independent | 612 | -- | 0.5016 |
