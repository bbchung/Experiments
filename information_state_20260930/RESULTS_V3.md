# V3 result: ordered repaired-demand chain

The frozen V3 contract is `68862b894c708852b8a62d5131aa05f0b310f3e580fe07f97521f1f4dd726858`.
Exactly three serialized CatBoost GPU fits completed. No candidate was nominated on calibration; neither candidate is supported on development. No pristine or sealed OOS was scored.

| Role | Arm | Joint AP | Magnitude AP | Conditional direction AUC |
|---|---|---:|---:|---:|
| calibration | baseline | 0.197889 | 0.404286 | 0.550377 |
| calibration | h10_ordered | 0.198980 | 0.404440 | 0.554129 |
| calibration | h10_reset | 0.200010 | 0.404429 | 0.558618 |
| development | baseline | 0.127281 | 0.262820 | 0.590404 |
| development | h10_ordered | 0.127847 | 0.263799 | 0.590710 |
| development | h10_reset | 0.126787 | 0.260884 | 0.583604 |

Ordered H10 misses calibration joint/magnitude/direction improvement gates. Its 30-day development paired-block lower bounds are -0.00189109 for joint AP, -0.00049771 for magnitude AP and -0.00477792 for conditional direction AUC. Equal-symbol-day magnitude and net directional-tail guards also fail. Small pooled point gains do not prove robust information.

The predeclared ordered-versus-reset ablation is also unsupported. The development direction lower bound is -0.00209975. This does not establish an advantage from retaining the repaired mark through touch renewal.

All 2,576 original cells passed exact keys, unchanged historical raw-mid bits, native state/type and the independently frozen receive-availability method. The historical/current whole-producer parity failure remains recorded; every arm retains the same original historical baseline values. The component and temporal integrity qualifications are not FE gain.

TRAIN-only phase diagnostics cover the exact 564,264 model origins. Ordered phases include 128,420 contained, 293,618 repaired, 7,538 renewed, 7,535 resumed, 126,750 censored and 403 stale. The uniquely resumed state appears in 1.335% of these origins. A lack of model gain is therefore not a claim that the source has no identifiable episodes.

Post-comparison model diagnostics show prediction-value-change importance 0 for all H10 fields except ordered mid_progress_5ticks (0.00147348 percentage points). Reset H10 has total importance 0. This is descriptive, not a revised judge: the fixed learner mostly ignored the new mechanism. The result rejects this complement under the present representation/training/sampling contract; it does not prove all ordered information useless at every horizon or in a different coherent representation.

The next source hypothesis is H13 quote-led current-visit bilateral testing, retained as historical accepted-price information, rather than prolonging incomplete short episodes. It must first survive fixed TRAIN support/identifiability and native causal validation. A separate compact three-role representation and conditional-direction methodology remain proposals; no old FE result is rejudged.

V1, V2 and V3 failures remain explicit. No feature is predictively admitted by these rounds, the legacy full baseline remains an unadmitted control with known raw-volume scales, and the active Goal is not achieved.
