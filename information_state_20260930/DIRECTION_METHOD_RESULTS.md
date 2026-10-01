# Conditional-direction methodology study results

The frozen study completed three prescribed conditional CatBoost fits and is
unsupported. The first calibration comparison passed, allowing the remaining
two fits. The three-repeat calibration ensemble failed the unchanged 2% joint
AP gain requirement, so it was not nominated. Development diagnostics show a
higher conditional direction AUC but worse 1% net directional tails and
nonpositive paired lower bounds. The method is not adopted; there is no FE
admission or pristine OOS claim.

This report and [direction-method-comparison.csv](direction-method-comparison.csv)
are post-run analysis artifacts outside every existing frozen source/input
closure. No source, profile, tests, feature values, models, sampling, labels or
acceptance rules were changed for this analysis. No additional fit or dataset
materialization was performed.

## Identity and measurement scope

The method contract is
`b06aab35c4a42859b6a477316ea848e2f778e5b611920fb37389beb05b73d4ef`;
its manifest contains 1,758 sources and 5,763 inputs. The fixed magnitude
control is the original V3 full baseline, selected by lifecycle identity
`68862b894c708852b8a62d5131aa05f0b310f3e580fe07f97521f1f4dd726858`.
MC2 and MC3 are the preregistered V2 and method1 full-baseline controls, not
controls selected using their scores. All comparisons use explicit fixed
`b=P(big)` and `q=P(up|big)`; joint rankings use `b*q` and `b*(1-q)`.

The primary endpoint is the native 300s mid-price return with inclusive +/-5
tick events. Every row remains mature at all four 60/120/180/300s horizons.
The original 30s evaluation origins, five-field keys, label hashes, universe,
chronological roles and held-symbol policy are unchanged. Calibration contains
106,284 rows over 20 days, 2026-07-14 through 2026-08-10. Development contains
221,280 rows over 30 days, 2026-08-12 through 2026-09-22. These dates were already
declared exposed development material. The future sealed OOS role remains
empty and requires at least 30 newly available days beginning 2026-10-01.

The conditional learner uses the original full-baseline 3,254 numeric and 132
categorical predictors. Original full-TRAIN weights are computed before the
native big-event subset and retain their original normalization. Receipts bind
34,180 event rows from 564,264 TRAIN rows and 3,143 event rows from 49,788 ES
rows; ES supplies early stopping only. The attempt ledger records exactly
repeats 1, 2 and 3. Each fit stopped at 98 trees under the fixed depth6,
600-iteration maximum, learning rate0.05, stopping50 and same seed. No feature
candidate or HPO entered this methodology study.

## Calibration decision

These are metrics of actual per-row score ensembles, not averages of scalar
metrics. The frozen effect requirements remain joint AP >=1.02*baseline and
conditional direction AUC >=baseline+0.01 and >=0.52; magnitude noninferiority
has a zero margin and additionally requires absolute power.

| Calibration comparison | Joint AP baseline | Joint AP Q | Relative joint gain | Direction AUC baseline | Direction AUC Q | Failed typed gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MC1 -> Q1, prescribed first gate | 0.197888892456 | 0.202271685635 | 2.2147747% | 0.550376759975 | 0.585539721214 | none |
| MC2 -> Q2, descriptive repeat | 0.199448846658 | 0.202271685635 | 1.4153198% | 0.548952847723 | 0.585539721214 | joint_gain |
| MC3 -> Q3, descriptive repeat | 0.199448860341 | 0.202271685635 | 1.4153128% | 0.548953074352 | 0.585539721214 | joint_gain |
| Actual MC mean -> actual Q mean, nomination | 0.198960730772 | 0.202271685635 | 1.6641248% | 0.549884974270 | 0.585539721214 | joint_gain |

The ensemble's required joint AP was 0.202939945387; the achieved value was
0.000668259752 below that requirement. Its direction gain was +0.035654746945,
and every other typed calibration gate passed. Calibration nomination is
therefore false. The first gate passing and the ensemble gate failing are
consistent with the prospective procedure; the control comparison is different
and MC controls vary while Q predictions do not.

The only acceptance tail gate uses the average of the two sides' top-1%
correct-event precision minus opposite-event precision. The following side
values make that aggregate explicit; neutral endpoints receive neither credit.

| Role, actual ensembles | Side | MC precision | MC opposite precision | MC net | Q precision | Q opposite precision | Q net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration | up | 0.291742424242 | 0.210909090909 | 0.080833333333 | 0.310075757576 | 0.205075757576 | 0.105000000000 |
| Calibration | down | 0.255000000000 | 0.226363636364 | 0.028636363636 | 0.254242424242 | 0.247803030303 | 0.006439393939 |
| Development | up | 0.160360360360 | 0.175675675676 | -0.015315315315 | 0.163963963964 | 0.181531531532 | -0.017567567568 |
| Development | down | 0.188288288288 | 0.150900900901 | 0.037387387387 | 0.181981981982 | 0.151801801802 | 0.030180180180 |

Calibration average net improves from 0.054734848485 to 0.055719696970,
so the 1% tail gate passes despite a weaker down side. The more extreme
calibration top-0.1% nets fall from 0.283333333333 to 0.150000000000 for up
and from -0.008333333333 to -0.050000000000 for down. Those 0.1% observations
are descriptive and do not introduce another acceptance criterion.

## Development diagnostics and failed support

| Actual ensemble metric | MC mean | Q mean | Difference |
| --- | ---: | ---: | ---: |
| Conditional direction AUC | 0.587921279431 | 0.613822372645 | +0.025901093214 |
| Joint mean AP | 0.127478248317 | 0.132318867522 | +0.004840619204 (+3.7972119%) |
| Equal-symbol/day conditional direction AUC | 0.552679018901 | 0.570218581116 | +0.017539562215 |
| Nonoverlap conditional direction AUC | 0.580135310356 | 0.619724569182 | +0.039589258825 |
| Nonoverlap joint side AP | 0.146353553361 | 0.146392150279 | +0.000038596918 |
| Average top-1% net directional precision | 0.011036036036 | 0.006306306306 | -0.004729729730 |
| Direct magnitude AP | 0.262819817284 | 0.262819817284 | exact zero |

The development ensemble's intrinsic failed gates are
`net_directional_tail_noninferiority`, `paired_mean_ap_gain`, and
`paired_conditional_direction_auc_gain`. The final result also fails the
previous-calibration nomination requirement and the requirement that all
three paired repeat decisions be supported. All absolute-power, equal
symbol/day, seen-symbol and held-label-symbol direction/magnitude guards pass;
they do not override the failures.

The frozen paired procedure uses identical chronological days, moving blocks
of 3 days, 2,048 resamples, seed20260930 and one-sided tail0.05. The table below
transcribes the persisted decision intervals, without recomputing or rounding
inputs to change the decision. Both claimed gain lower bounds must be strictly
positive. Magnitude lower/mean/upper are exactly zero for every comparison.

| Development comparison | Daily mean joint AP delta | Joint AP lower bound | Daily mean direction AUC delta | Direction AUC lower bound |
| --- | ---: | ---: | ---: | ---: |
| MC1 -> Q1 | +0.003153781542 | -0.001370109690 | +0.017216256810 | -0.002997029419 |
| MC2 -> Q2 | +0.003139726614 | -0.001256414504 | +0.020801524131 | -0.006458647669 |
| MC3 -> Q3 | +0.003139589272 | -0.001256647985 | +0.020798355864 | -0.006459756562 |
| MC mean -> Q mean | +0.003194202971 | -0.001026857121 | +0.018644136503 | -0.005927747751 |

All three repeat decisions fail the tail and both paired-gain guards.
Repeats2/3 additionally fail nonoverlap joint noninferiority: Q side AP
0.146392150279 is below MC2's 0.146515721624 and MC3's 0.146515539647.
For the actual ensemble, direction improves on 19 days, ties on 1 and declines
on 10; joint AP improves on 15 and declines on 15. These descriptive counts
explain why pooled improvement does not establish a positive paired bound.

The held-label group has 55,320 development rows, direction AUC
0.570498258367 -> 0.616983538951 and joint AP
0.103777328466 -> 0.109524467390. Seen symbols have 165,960 rows, direction AUC
0.587790187610 -> 0.605750923708 and joint AP
0.134218401599 -> 0.138616599173. No symbol was selected or excluded using
these outcomes. Held symbols remain the preregistered 2317, 2615, 2409 and 3711,
whose labels were excluded from TRAIN, ES and calibration.

## Repeats, true ensembles and ordered null comparisons

Q1/Q2/Q3 are bit-identical probability vectors on both scored roles. Their
direction AUC ranges are exactly zero: 0.585539721214 on calibration and
0.613822372645 on development. Model file hashes differ, so this is prediction
equality on these cohorts rather than a claim of byte-identical model files or
general GPU determinism. MC direction AUC ranges are 0.001423912253 on
calibration and 0.004882027962 on development. The recorded paired repeat
direction deltas, [0.035162961239, 0.036586873492, 0.036586646862] on calibration
and [0.023418231791, 0.028300259754, 0.028295704044] on development, reflect
variation in the existing MC controls; they are not Q repeat variation.

Saved three-repeat mean q arrays match the exact prospective `np.mean` operation
bit for bit. Although Q repeats are identical, summation/division changes
10,608 calibration and 24,206 development mean-q float64 bit patterns relative
to Q1. These are arithmetic differences; all reported ranking and daily
metrics remain identical. The saved Q mean is the actual ensemble and is not
replaced by Q1 for evaluation. Actual two-repeat leave-one-out Q ensemble
metrics are also identical to Q1. The MC leave-one-out direction AUC values
are 0.585523935175, 0.588811595139 and 0.588810163344; their joint AP values are
0.127442636699, 0.127463358622 and 0.127463319446. No leave-one-out ensemble was
selected.

| Ordered development null | Direction daily mean delta | Direction lower bound | Joint AP lower bound | Supported |
| --- | ---: | ---: | ---: | --- |
| MC1 -> MC2 | -0.003585267321 | -0.010544598561 | -0.001757459722 | false |
| MC1 -> MC3 | -0.003582099054 | -0.010539212508 | -0.001757054563 | false |
| MC2 -> MC1 | +0.003585267321 | -0.006690065464 | -0.002133014741 | false |
| MC2 -> MC3 | +0.000003168267 | 0.000000000000 | 0.000000000000 | false |
| MC3 -> MC1 | +0.003582099054 | -0.006697208120 | -0.002133104014 | false |
| MC3 -> MC2 | -0.000003168267 | -0.000009504800 | -0.000000412026 | false |
| Q1 -> Q2 | exact zero | exact zero | exact zero | false |
| Q1 -> Q3 | exact zero | exact zero | exact zero | false |
| Q2 -> Q1 | exact zero | exact zero | exact zero | false |
| Q2 -> Q3 | exact zero | exact zero | exact zero | false |
| Q3 -> Q1 | exact zero | exact zero | exact zero | false |
| Q3 -> Q2 | exact zero | exact zero | exact zero | false |

All 12 null pairs reject, so that guard passes. Three same-seed fits and
correlated existing controls do not estimate general FPR, seed variability or
future repeatability. The identical Q predictions provide no independent
ensemble diversification on these cohorts.

## Exact magnitude and procedural checks

Both saved b arrays are float64; their uint64 artifacts exactly match the
original V3 stored up+down class-probability sum and the persisted bit hashes.
The float64 and uint64 file SHA receipts also match. Every one of the 8
calibration and 14 development metric reports has the identical role-specific
five-key/four-label signature and b bit hash. The magnitude AP/base rate,
equal-symbol/day magnitude AUC, absolute-move Spearman, nonoverlap magnitude AP
and seen/held group magnitude metrics match exact float64 bits across all
reports. All daily magnitude AP values and day axes also match exact bits
(20 calibration days; 30 development days). Every stored joint score array
exactly equals its saved `b*q`/`b*(1-q)` components; magnitude is never inferred
from their potentially rounded sum.

| Role | b bit SHA256 | Direct magnitude AP | Event base rate | Equal-symbol/day magnitude AUC | Absolute-move Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | e5b34c461f2cdb5b93e6a913c23f840d3fdde4168fa53c600b76783102fa8c64 | 0.404285718148 | 0.055916224455 | 0.785417866183 | 0.559017781270 |
| Development | 0ec791117cd420b71223ef66bfe9f20e2e11282df35d1111af13609fea1cbeeb | 0.262819817284 | 0.025415762834 | 0.789433500206 | 0.478671022010 |

The direct magnitude development bit hash in this table is checked against
`scores/fixed-b-development.yaml`; the score and uint64 artifact receipts are
the authoritative identities. Exact fixed magnitude establishes nuisance
equality and absolute power, not a magnitude gain.

Development diagnostics after the false nomination follow the prospective
plan. [DIRECTION_METHOD_PLAN.md](DIRECTION_METHOD_PLAN.md) explicitly reports
all fitted repeats and both ensembles on development, including failures,
after all calibration reports and nomination are persisted. The frozen runner
writes nomination before its first development material load and always
requires `nominated_before_development` for support. Artifact timestamps agree:
the last calibration metric precedes nomination, which precedes every
development metric. Thus development is diagnostic exposure, not nomination
rescue or a replacement acceptance path. First-calibration failure would have
prevented development access; that early-rejection condition did not occur.

## What the result teaches and what it does not establish

Changing only the conditional training responsibility extracts more pooled
direction information from the existing baseline inputs. The gain reaches
equal-symbol/day and nonoverlap direction metrics and both fixed symbol
groups, while meaningful top-tail correctness and paired day stability remain
insufficient. Consequently the evidence is broader ranking improvement under
this practical training formulation, rather than a stable admitted improvement.
`L_MC=L_big+1_big*L_direction` still rules out an inherent mathematical
incompatibility of magnitude and direction. Big-event subsetting also changes
internal quantization and categorical target statistics, so finite capacity or
early stopping has not been isolated as the causal explanation.

The preceding full MultiClass learner's zero or tiny H10 importance cannot
establish that its underlying information source is absent: this study changes
extraction from the unchanged baseline and obtains measurable direction
differences. However, it adds no H10 or H13 input, performs no new MI screen or
feature ablation and measures no conditional-head feature importance. It
therefore neither validates a sparse mechanism nor proves that a previous
low-MI/zero-importance feature would become useful with conditional training.
Sparse support, overlapping episodes, informative missing states and an
absence of independent events remain possible measurement limitations, to be
tested using source support and frozen representation comparisons. The legacy
full baseline's raw-scale normalization issues also remain; this method study
cannot admit a complete cross-symbol-normalized replacement.

Two information hypotheses merit later source reasoning, after the current
native H13 gate, rather than relaxing this study's gains or tails:

1. **Contradiction and failed acceptance.** Represent whether an apparent
   signed advance is challenged by confident opposite-side execution and
   replenished resting supply before a completed price-visit acceptance. A
   failed/contested visit state can retain sign and failure cause instead of
   pooling it with successful demand. This directly addresses the case where
   pooled direction ranking improves but top-1% opposite events increase.
   Normalize causal known-side work by same-visit opposing work or observed
   relevant depth, price progress by five ticks, and each opposing-refill
   component by its own observed supply. Preserve unknown/warmup/censored
   states; no epsilon denominators or raw volume ranking. H13's fixed native
   bilateral acceptance gate is the immediate source test; a separate failure
   representation would need its own prior support test and could be rejected
   if failed visits have no repeatable distinct conditional information.
2. **Independent evidence concentration.** Represent whether directional
   support consists of many updates from one closed exchange cluster/visit or
   corroborating known work from distinct causally closed clusters, together
   with a counter-evidence share. This tests whether high scores attach to
   repeated but dependent evidence. Use proportions of same-episode observed
   work and concentration over actual evidence units, rather than unnormalized
   trade frequency or a longer rolling window. Reject the hypothesis if a
   bounded TRAIN-only support audit finds too few independent units or if the
   newly frozen comparison cannot improve joint tails and paired evidence.

These are hypotheses, not observed explanations of the errors. Any material
feature implementation must be a native C++ FeatureModule in the experimental
feature folder with ValueWriter output. Python may evaluate that material;
it must not recreate the representation. A future FE comparison starts from
a fresh baseline under an already frozen judge. The conditional study's
unsupported method is not silently promoted to that judge.

## Artifact trace

All relative run paths below resolve under
`../../runs/information_state_20260930/direction-method/`:

- `frozen-contract.yaml` and `.identity`: preregistered scientific closure.
- `fits/q{1,2,3}.yaml`, `fit-attempts.yaml`: exact three-fit budget, TRAIN/ES
  binding, original subset-weight hash and 98-tree receipts.
- `first-calibration-gate.yaml`, `nomination.yaml`: prescribed first-gate pass
  and ensemble nomination failure.
- `scores/{mc1,mc2,mc3,mc-mean,q1,q2,q3,q-mean}-{role}-metrics.yaml` and
  `-daily.csv`: main comparisons; `*-q.npy` and `*-joint.npy`: actual vectors.
- `scores/fixed-b-{role}.{npy,yaml}` and `-bits.npy`: magnitude bit/file receipts.
- `development-decisions.yaml`: official ensemble/repeat gates and paired bounds.
- `null-pairs.yaml`, `stability.yaml`, `scores/*leave-out*`: bounded repeat checks.
- `completed.yaml`: unsupported, method_adopted=false, feature_admission=false,
  pristine_oos=false, development_read=true.

The CSV contains the 16 main role/report rows and 6 descriptive development
leave-one-out rows. It transcribes existing metrics; conditional-repeat
comparisons use their preregistered MC counterpart, while mean comparisons use
the actual MC mean. It includes persisted development paired lower bounds and
failed-gate names and does not rank or select any alternative model.
