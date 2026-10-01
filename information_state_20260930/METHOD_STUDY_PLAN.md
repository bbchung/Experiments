# Independent claim-conformance and training-repeat method study

Status: methodology research proposal/prototype, separate from FE comparison.
V1 and V2 remain immutable. No V2 challenger outcome determines these rules.
Toy fixtures establish mathematical semantic conformance, not predictive gains
or empirical superiority of a replacement methodology. No new FE comparison
may use this prototype before its methodology is validated and frozen in an
official profile/kernel with a fresh baseline comparison.

## Independent questions

1. Can a direction-only claim improve conditional direction and joint event
   ranking while preserving an already useful magnitude component exactly?
2. Can a magnitude-only claim improve large-event detection and joint ranking
   while preserving an already useful direction component exactly?
3. Does identical CatBoost training produce enough variation to change a
   research decision, and does a fixed repeat-aggregation procedure reduce that
   decision's sensitivity without selecting a lucky training realization?

The first two follow directly from the user's permission for direction-only
and magnitude-only features while requiring the final representation to have
both kinds of predictive power. V1 requires every candidate to improve both
components. In particular, fixed `P(big)` implies identical magnitude AP, so
the magnitude-gain gate and positive magnitude daily bound cannot be met by
a direction-only improvement, however strong its direction/joint improvement.

GPU CatBoost can vary across repeated training because the order of floating
point summations varies; fixing the seed does not guarantee identical GPU
training. This mechanism is documented in the official
[CatBoost FAQ](https://catboost.ai/docs/en/concepts/faq).

## Controlled semantic study

Use mathematical score fixtures with explicit heads:

```
p_up = p_big * q_up_given_big
p_down = p_big * (1 - q_up_given_big)
```

The fixtures are unit-test controls, not synthesized missing market data.
Hold one head bitwise unchanged while improving the other's ranking. Native
mid-price endpoint label semantics and all legacy effect-size controls remain
the same. The tests also verify that unchanged scores cannot qualify, improving
one component cannot excuse worsening the other, missing evidence is not
dropped, and a both-components claim really requires both gains.

Retain the meaningful frozen controls for the comparison: joint AP must gain
at least `acceptance.relative_ap_gain` (currently 2%); a magnitude claim must
gain the same 2%; a direction claim must gain
`acceptance.direction_auc_gain` (currently one percentage point). Every final
representation retains `minimum_direction_auc` (currently 0.52). These controls
are inherited unchanged to isolate the claim-semantics change; they are not
chosen from challenger results or presented as universal economic thresholds.

Declare each hypothesis's claim before fitting: `direction`, `magnitude`, or
`both`. An unclaimed component requires point noninferiority with margin exactly
zero. Its paired daily lower bound must be at least zero instead of strictly
positive. Claimed components and joint ranking still require positive paired
daily bounds. Keep equal-symbol-day, nonoverlap, event-support and seen/held
symbol-group checks. Require magnitude AP above the event base rate and
equal-symbol-day AUC above chance; pooled direction keeps its legacy floor.
Do not reinterpret absolute-amplitude regression claims as large-event
probability claims: this prototype covers the latter only.

No margin is fitted to FE results. Nonzero noninferiority margins are rejected
by the prototype. Any later proposal for a nonzero resolution margin must be
derived in a separately frozen, independent TRAIN-only variability study; it
cannot use calibration/development performance to relax an acceptance gate.

Component metrics consume explicit `p_big` and `q`, not their reconstructed
joint sum or ratio. The controlled fixture shows that identical `p_big=0.1`
can produce distinct joint sums for different q by one ULP, changing tied
magnitude AP from 0.5 to 1.0. This is a score-contract issue in a factorized
method; it does not prove an existing native-label or V1/V2 prediction bug.
Joint products define joint ranking; the exact declared heads define component
ranking. Zero, invalid or underflowed joint probabilities are rejected without
epsilon repair.

## Bounded prospective null study

Use three identical-representation, identical-parameter, SAME-SEED baseline
training realizations:

- Reuse the immutable V1 full-baseline model/receipts and calibration/development
  scores.
- Reuse the immutable V2 fresh full-baseline model/receipts and the same scores.
- Root alone performs one newly reserved full-baseline 300s fit after V2 is
  terminal, with the same native cohort, feature/nominal order, weights, early
  stopping role, parameters and seed. Score the existing calibration and
  already-exposed development origins.

Before the prospective fit, freeze the method-study profile, this plan,
prototype/tests, inherited scientific sources, two existing model/receipt/score
identities, native input identities and future third-fit recipe. GPU fits remain
serial under the global GPU0 lock. No additional feature arm, HPO, materialization
or horizon search enters this study. A single fixed seed across these three fits
measures within-seed training nondeterminism; it is NOT evidence for multi-seed
robustness.

Verify before scoring: all role/row-key/label hashes agree, model feature/nominal
order and classes agree, package/environment identities agree, and training
formulation and parameters agree. Full freeze IDs may differ because experiment
identity and arms differ; cohort payload and substantive judge must match.

Predeclared analyses:

1. Report each realization's pooled, daily, equal-symbol-day, nonoverlap and
   seen/held-group magnitude, direction and joint metrics without selecting
   the best realization.
2. Evaluate all six directed baseline-to-baseline pairs under legacy gates and
   each predeclared claim prototype, separately for calibration nomination
   and development qualification. Calibration is not final evidence.
3. Report leave-one-out two-realization means against the three-realization
   metric mean, plus decision sensitivity. Average corresponding scalar/daily
   metrics across realizations; this is not a prediction ensemble. Use the same
   chronological day axis and block draws throughout.
4. An optional TRAIN-only score probe may describe training variability if root
   can generate it from a predeclared sample without another fit. In-sample
   probe results are not OOS evidence and do not authorize nonzero margins.

Gate declared before the new fit: none of the six identical-representation
null pairs may receive a supported development decision under the prototype.
Any such decision demonstrates inadequate resolution for claiming an FE gain
from a single training realization; report the method as unqualified rather
than changing margins, deleting days, or selecting a better repeat. Compare
nomination/qualification sensitivity with the legacy method. Improvement in
semantic conformance must not be accompanied by observed stability regression.
If both methods reject every null comparison and show the same sensitivity,
report equivalent observed stability, not empirical stability superiority.

Three fits and six correlated pairs cannot certify a population false-positive
rate or general seed robustness. Small-sample limitations remain explicit.
Resampling these three fits cannot create additional independent training
realizations. All historical development results remain exposed; the sealed
future OOS requirement of at least 30 new dates remains untouched and currently
unimplemented.

## Possible subsequent formulation study

A separately frozen CatBoost-only two-stage formulation could share an
unchanged magnitude head across baseline/candidate representations and fit
claim-specific conditional direction heads on native large-event training
labels. This preserves nuisance magnitude by construction and fits the strict
zero-margin rule. It requires its own baseline-to-baseline formulation study
before any FE comparison; it is not part of this one-fit null study.

Likewise, a prospective paired-seed mean procedure would fit baseline and every
candidate at the same predeclared seeds and aggregate corresponding daily
metrics. Existing same-seed fits cannot validate that procedure. Its extra
compute and any change in training formulation must be frozen before use.

## Artifacts and current validation

Prototype API in `astra/representation_methodology/judgement.py`:

```
claim_decision(profile, baseline, candidate, baseline_daily=None,
               candidate_daily=None, *, claim, stage="calibration")
factorized_metrics(rows, p_big, q, horizon)
factorized_daily(rows, p_big, q, horizon)
```

Run semantic validation with:

```bash
python3.13 -m unittest tests.test_representation_methodology -v
```

Eight tests pass. They prove the stated mathematical/control behavior and
rejection contracts. Prospective null-fit results, repeat stability and a
production replacement remain unproven until the separate study completes.
