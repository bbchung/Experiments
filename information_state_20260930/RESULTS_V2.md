# V2 results: full-context complement and marked renewal rejected

Frozen identity: `6eb02ef89e592bf97bb42b9c30b5abce0f508e5cd50553556a014ccbc43cd033`.

All ten predeclared CatBoost fits completed under the unchanged V1 judge and exact native cohort. No calibration nominee, no accepted representation and no sealed OOS scoring. V2 tests whether the prior state families complement the full baseline and whether price-renewal arrival, dwell, mark and direction state contribute new information.

| 300s arm | calibration joint AP | calibration big AP | calibration direction AUC | development joint AP | development big AP | development direction AUC |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.199844 | 0.404979 | 0.548953 | 0.127601 | 0.264158 | 0.585522 |
| old_context | 0.197898 | 0.404157 | 0.556805 | 0.132076 | 0.263507 | 0.599653 |
| renewal | 0.192460 | 0.403251 | 0.530701 | 0.127332 | 0.258923 | 0.589159 |
| core | 0.182947 | 0.387540 | 0.509353 | 0.124661 | 0.251157 | 0.581635 |

`old_context` preserves the full native context and adds graph, session and occupation states; `renewal` adds H9 to the full native context; `core` adds H9 to the compact semantically scaled core. None meets the frozen calibration gates. Development scores cannot rescue an un-nominated arm.

The old-context arm's development pooled joint AP rises 3.5% and direction AUC rises 1.41 points, while magnitude AP declines. The paired daily lower bounds are negative for joint AP (-0.001404), magnitude AP (-0.002430) and conditional direction AUC (-0.017078). Seen and label-held-out magnitude checks also fail. These are insufficient evidence of a generalizable gain; the judge and rejection remain unchanged.

H9 passes causal prefix, grid and missing-state checks but fails predictive nomination. All 3,540,544 native mids satisfy the strict physical half-cent lattice. The independent native-label boundary census finds no zero/nonzero or inclusive five-tick classification disagreements. It covers 14,161,120 finite endpoints, including 114,525 exact +/-5 labels. No label repair or threshold tolerance was introduced.

The same baseline recipe produces different GPU training realizations across V1 and V2: calibration direction AUC differs by 0.00968, close to the fixed 0.01 gain threshold. This motivates a separately preregistered methodology study. It does not alter either completed FE comparison. Identical saved model artifacts remain reproducibly scoreable; three same-seed realizations cannot establish general seed robustness.

The next native hypothesis links known executions to delayed quote-only repair at an original anchor, a subsequent distinct touch, and renewed same-side execution at that new price. This chronology is not recoverable from a longer rolling return window. Its TRAIN-only event support and native causal implementation must pass before a new frozen FE comparison.

Artifacts: `runs/information_state_20260930/v2/comparison.csv`, `nomination.yaml`, `development-decisions.yaml`, `shared/renewal-prefix-validation.yaml`, `shared/native-label-boundary-census.json` and `fits/*-metrics.yaml`. Current evaluation dates were historically exposed; they remain development evidence. Final OOS still requires at least 30 previously unobserved dates beginning no earlier than 20261001.
