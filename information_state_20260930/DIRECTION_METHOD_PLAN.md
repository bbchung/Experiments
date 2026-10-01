# Prospective fixed-magnitude conditional-direction study

Status: new, unexecuted and unfrozen methodology recipe. No actual dataset or
scored calibration/development arrays were read while implementing this study;
only immutable profile/manifest/fit-receipt metadata supplied bindings. Root
alone may freeze and run the new method. Nothing is formally adopted and no
native FeatureModule or feature transformation is implemented by this package.

The hypothesis is a finite-capacity/early-stopping limitation of MultiClass's
shared representation of big-event probability and conditional direction.
`L_MC = L_big + 1_big * L_direction` excludes a claim of inherent mathematical
incompatibility. A separate CatBoost Logloss conditional head must demonstrate
meaningful reproducible direction and joint-ranking improvement before the
method can be considered useful. No HPO, alternative model or feature candidate
enters this study.
Training on the big-event subset also changes the population supporting
CatBoost's internal quantization and categorical target statistics. The same
frozen model parameters do not make these learned internals identical. A gain
would support the practical conditional-training formulation; it would not
isolate shared capacity or early stopping as its causal explanation. No
additional quantization search or representation transform is performed.

The first fixed control is V3's original full-baseline model under contract
68862b894c708852b8a62d5131aa05f0b310f3e580fe07f97521f1f4dd726858.
V2 baseline and method1's prospective null baseline are the other two existing
same-seed controls, selected by artifact lifecycle rather than measured scores.
Each control's frozen manifest/anchor/profile, model, receipt and original
role-specific score/metric files is bound. Its model schema, TRAIN/ES five-key
and four-label signatures, original weights, loss/parameters and feature order
must match. Any mismatch rejects the study; no replacement control or dataset
may be selected after results.

On first use of each evaluation role, derive `b = original_up + original_down`
exactly once from the fixed V3 float64 score array and save its uint64-bit
identity. Other control q values use their own original up/(up+down), but every
control and challenger uses the same fixed b. Explicit b defines magnitude
metrics and explicit q defines conditional direction. Joint ranking uses b*q
and b*(1-q). A zero/invalid magnitude or nonfinite/out-of-domain head rejects;
there is no epsilon repair, clipping, magnitude resummation or row filtering.
The float64 b artifact and a separate uint64-bit artifact are both persisted
with hashes and checked on resume. Magnitude metrics/daily values compare exact
float bits, including undefined-state identity; required undefined evidence
cannot qualify the method.
Saved b and all reports preserve exact role/cohort/label provenance. Future
leakage rejection is structural: frozen source/input closure, original feature
names excluding labels, fitting/stopping roles, class order and receipt hashes.
Metrics alone cannot prove a supplied score's causal origin.

All original sampling, native endpoint +/-5 labels, complete four horizons,
TRAIN/ES/calibration/development dates, universe/held-label rules, numeric and
nominal features, recency weighting and hyperparameters remain unchanged.
Only the new conditional learner's target support and loss differ. Compute
weights on the full known TRAIN population exactly as the baseline, then
subset to native big events, preserving original normalization and symbol/day
denominators. Direction labels are down=0, up=1. ES only subsets to native big
events and remains unweighted, matching original ES weighting. It controls
early stopping only; no calibrator, threshold, fusion learner or feature selector
fits on ES/calibration/development. Train and ES require both direction classes.

Create the conditional TRAIN/ES Pools once and reuse them. There are at most
three identical-seed Logloss fits, depth6/600iterations/0.05learning-rate with
the original remaining settings and stopping50. Every fit holds the single
global GPU0 lock. A durable attempt ledger reserves each numbered fit before
training; failed/orphaned attempts cannot silently restart and exceed budget.
Model resume requires exact frozen digest, schema, masks, weights, classes and
model hash. No duplicate baseline cache/dataset or simultaneous GPU process is
created by this study.

The first challenger calibration report is compared with fixed control zero.
Failure of any typed calibration gate early-rejects after at most one new fit,
without reading any development dataset/labels/scores. Otherwise complete all
three fits and all calibration reports. The actual per-row mean-q challenger
is compared with the actual mean-q MC control and nomination is written before
the first development material read. All fitted repeats and both ensembles are
reported on development, including failures. No best-repeat selection occurs.

The typed direction rule reuses the frozen claim prototype's effect sizes:
joint AP >=baseline*1.02; conditional direction AUC >=baseline+0.01 and >=0.52.
Magnitude AP, equal-symbol/day magnitude, absolute movement ranking and daily
magnitude values are identical because b is fixed; no gain is claimed for that
component. Final representation must still have absolute magnitude and direction
power. The claim prototype retains equal-symbol/day power/noninferiority,
directional tail, nonoverlap joint ranking, same-day paired block bounds with
block3/draws2048/tail0.05, support>=100per-side on>=15days, and seen/held-symbol
power/noninferiority. Undefined required evidence or omitted groups rejects.
The final method support additionally requires prior calibration nomination,
all three paired repeat development decisions, and no supported ordered null
comparison among the three MC repeats or among the three Q repeats.

Persist all twelve within-formulation ordered development null decisions and
paired replica direction deltas. Score each actual leave-one-out two-model q
ensemble on the same development rows; these are descriptive stability results
and cannot be selected. Three same-seed GPU repeats establish only bounded
repeatability; they do not identify seed variance or calibrate a general FPR.
The null comparisons use the same b, eliminating nuisance magnitude noise.

The complete method source closure includes inherited evaluation/data/scientific
sources, all control manifests' dependencies, the new kernel/profile/plan/outline
and semantic/orchestration tests. Preflight validates immutable model/procedure
metadata without reading datasets or scores. Freeze hashes the exact inputs
before any new fit. Verification reconstructs mandatory dependency membership
so rewriting a canonical manifest/anchor cannot drop controls, scores or kernel.
Role-specific score cohort signatures are rechecked before use. All calibration
scores precede any development scoring. Training/calibration failures cannot
retain a stale completion/nomination.

Controlled tests prove a meaningful Q-only improvement at fixed b is accepted,
while identical scores, nuisance drift, direction harm, weak support, omitted
groups, nonpositive paired evidence, wrong roles/keys/labels, unsafe head states
and closure/resume/fit-budget drift are rejected. They are not synthetic market
evidence or empirical methodology superiority. Real study failure leaves the
existing judges intact. Even a successful study reports method_adopted=false,
feature_admission=false and pristine_oos=false. The raw-scale historical full
baseline remains a qualified control; a complete normalized FE replacement and
newly sealed future OOS evidence remain separate required research.
