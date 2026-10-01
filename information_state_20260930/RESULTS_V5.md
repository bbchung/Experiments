# V5: compact native context and conditional response results

V5 rejects both compact replacements. Calibration nominated neither C15 nor
C18; the completed development diagnostics do not create a later nominee. The
compact representations lose magnitude discrimination and joint side ranking
against a fresh legacy baseline. Three conditional FlowResponse fields are used
by CatBoost, but their predeclared C18-versus-C15 ablation does not establish an
improvement. These results do not prove that the fields or their information
source are intrinsically useless.

This report was written after the three authorized primary fits completed. It
changes no feature, frozen source, profile, judge, sample, symbol universe or
training parameter. Analysis reads the existing small metric/decision/fit
receipts and, separately, TRAIN-derived importance stored in the three CBM
models. No feature matrix, score array, raw tape or additional label values were
read; no extra fit or model tuning was performed.

## Contract and actual execution

- Frozen V5 identity: `14eb774fac9647f7da2e8da1fa16dac67cfa0f80b2b4ec55fe13086d869425f5`.
- Completed pipeline manifest digest: `2d7375f0c76d3b5b11009d91901a6dff0e42d429092df160da4796affd6fd86e`.
- Actual evidence root: `runs/information_state_20260930/v5/`.
- `completed.yaml`: three primary fits, zero secondary fits, nomination null,
  supported false, pristine OOS false, sealed scored false.

The unchanged frozen V3 primary-300s judge evaluates pure native MID +/-5-tick
events. Original 30s origins, chronological roles, all-four-horizon label
completeness, class weights, CatBoost MultiClass formulation, hyperparameters,
early stopping, nonoverlap cohorts, paired daily bounds and seen/held group
guards are shared by all arms. Representation availability never selects rows.
The actual comparison signatures match across all three arms separately on
calibration and development. The 60/120/180s labels constrain cohort completeness; no fit or effectiveness
claim at those horizons follows from this primary-300s comparison.

| Arm | Original native predictors | TRAIN rows | Retained trees | Fit seconds |
| --- | --- | ---: | ---: | ---: |
| B | Legacy selected 3254 numeric +132 categorical | 564264 | 443 | 22.674 |
| C15 | Audited physical/activity/anchor/supply context, 15 numeric +2 categorical | 564264 | 591 | 5.510 |
| C18 | C15 plus three native conditional FlowResponse fields, 18 numeric +2 categorical | 564264 | 600 | 5.465 |

The compact arms reduce fit cost substantially within this fixed experiment,
but compute savings do not meet the predictive criterion. Different retained
tree counts are outcomes of the same early-stopping procedure. C18 reaches the
fixed 600-tree budget; that is an interpretation limit, not evidence that a
larger budget would repair the representation or authorization for tuning.

All three calibration scores preceded persisted nomination. Development
diagnostics for all arms despite no nomination were explicitly authorized in
`V5_PLAN.md`. The plan's preserved prospective status text does not replace the
actual frozen contract and completion receipt. Development is already exposed
research material, not a pristine future OOS test. The raw-scale legacy feature
universe remains a scientific control, not a normalized representation admitted
for multi-symbol use.

## Magnitude and direction separate under the same judge

Calibration has 106284 rows and a large-move base rate of 5.5916%; development
has 221280 rows over 30 days and a base rate of 2.5416%. This change in event
prevalence/context limits transfer of pooled metrics across roles. It does not
make within-role, same-cohort comparisons unfair.

| Role / arm | Joint side-mean AP | Large-move AP | Direction AUC given large move | Absolute-move Spearman | Equal-symbol/day large-move AUC | Equal-symbol/day direction AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration B | .199844 | .404979 | .548955 | .548273 | .784538 | .538183 |
| Calibration C15 | .190352 | .396704 | .565512 | .525850 | .766442 | .488903 |
| Calibration C18 | .193232 | .397686 | .568172 | .526479 | .768700 | .519934 |
| Development B | .127600 | .264157 | .585516 | .478229 | .790296 | .557742 |
| Development C15 | .124727 | .243368 | .592636 | .457433 | .778078 | .519231 |
| Development C18 | .125185 | .242034 | .591566 | .458596 | .779975 | .519561 |

Against B on development, C15 joint AP falls 2.25% and large-move AP falls 7.87%;
C18 joint AP falls 1.89% and large-move AP falls 8.38%. Absolute-move ranking and
within-symbol/day magnitude discrimination also deteriorate. This is a broader
magnitude failure than a single chosen metric.

Pooled direction AUC rises by .007120 for C15 and .006050 for C18, below the
frozen +.01 development gain requirement. More decisively, equal-symbol/day
direction AUC falls from .557742 to approximately .519. The daily direction
differences below are negative on average. Pooled directional improvement
therefore cannot be called robust local or cross-symbol directional improvement.
Composition, heterogeneous difficulty or lost local conditioning are plausible
explanations, not identified causes.

Both compact arms fail calibration joint gain, magnitude gain, net directional
tail, nonoverlap ranking and within-symbol/day magnitude checks. Calibration
direction gain/power alone cannot qualify a complete representation. On
development both still fail joint/magnitude/direction gain, nonoverlap ranking,
within-symbol/day magnitude, all three positive daily bounds and both seen/held
magnitude generalization checks. No post-development choice is authorized.

The nonoverlap cohort gives the same magnitude/joint conclusion: development
large-move AP is .287652/.274669/.274450 for B/C15/C18, and side-mean AP is
.147198/.143824/.141075. Direction AUC improves to .603679/.596526 from .575806,
but this isolated direction result does not repair the complete representation.
Development top-1% correct-minus-opposite side precision is
.012580/.015090/.019144; calibration is .057917/-.001970/.003295. These are
pure-signal tail diagnostics, not trading returns or an alternate acceptance
rule.

## Paired daily evidence and symbol groups

The frozen comparison uses 2048 moving three-day-block bootstrap resamples over
30 development days, with one-sided tail probability .05 and sufficient frozen
resolution. The following are equal-day mean differences and the reported
intervals; they are not pooled differences, independent-event confidence
intervals or future performance guarantees.

| Contrast | Joint AP difference [lower, upper] | Large-move AP difference [lower, upper] | Direction AUC difference [lower, upper] |
| --- | ---: | ---: | ---: |
| C15 - B | -.007488 [-.014876, -.001161] | -.017114 [-.027468, -.010718] | -.010473 [-.047504, .024347] |
| C18 - B | -.008498 [-.015592, -.001711] | -.017709 [-.028532, -.011037] | -.012431 [-.046941, .022255] |
| C18 - C15 | -.001010 [-.002220, .001170] | -.000595 [-.002513, .001503] | -.001958 [-.008034, .008140] |

For both replacements the upper bounds of joint and magnitude differences are
negative. Failure is not merely a positive point estimate without sufficient
power. Direction is inconclusive under these paired bounds and does not meet a
positive lower-bound requirement.

| Development group / arm | Rows | Joint AP | Large-move AP | Pooled direction AUC | Equal-symbol/day direction AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seen B | 165960 | .134295 | .273587 | .586625 | .552890 |
| Seen C15 | 165960 | .133444 | .253945 | .589532 | .516857 |
| Seen C18 | 165960 | .134556 | .252410 | .586846 | .517568 |
| Held B | 55320 | .103850 | .234479 | .562542 | .571352 |
| Held C15 | 55320 | .100789 | .205527 | .600525 | .525889 |
| Held C18 | 55320 | .100664 | .204913 | .605576 | .525151 |

Magnitude loses within both groups, so the failure cannot be attributed only to
unseen symbols. C18's tiny seen-group joint advantage does not overcome its
magnitude loss, held-group failure or frozen full-cohort decision. The held
pooled-direction gain also coexists with much worse equal-symbol/day direction.
No group or symbol is selected after seeing these diagnostics.

## Independent FlowResponse ablation

Only the three original 300s-reference fields distinguish C18 from C15:
`flow_surprise0.0`, `response_innovation0.0`, and `response_coupling0.0` from
`FlowResponseSurprise.0`. Shared columns, categories and cohorts are identical.
The existing native formulas use prior joint capacity-relative flow and
tick/second response observations; this is not new Python feature arithmetic.
Original producer-vintage NaN/-inf states remain unchanged.

C18 versus C15 has calibration joint AP +.002880 (+1.51%), large-move AP +.000982
(+0.25%), and direction AUC +.002660. None reaches its frozen gain requirement.
Development joint AP improves only .000458 (+0.37%), while large-move AP falls
.001335 and direction AUC falls .001070. All three paired intervals cross zero.
The independent `c18-versus-c15.yaml` decision is unsupported on both roles.
Passing several absolute-power/tail/context checks is not compounded predictive
gain. Even a successful C18-versus-C15 comparison would not establish superiority
to B; here neither comparison passes.

## TRAIN-only model usage and bounded failure attribution

Model importance was obtained from the saved CBM models with
`get_feature_importance(type="PredictionValuesChange", thread_count=1)`, using
stored TRAIN leaf information, no supplied data and no refit. This multiclass,
correlation-sensitive split-use measure does not allocate direction versus
magnitude information, establish causal effects or authorize selecting columns
after development outcomes.

| Family / group | B importance % | C15 importance % | C18 importance % |
| --- | ---: | ---: | ---: |
| MaturedPathRegime | 8.365 | absent | absent |
| IntradayRegime | 8.175 | 51.191 | 47.528 |
| CauseConditionedQueueResponse | 6.683 | absent | absent |
| MaturedBarrierOutcome | 6.271 | absent | absent |
| AuctionCarry | 5.841 | absent | absent |
| LiquidityTransmissionState | 5.398 | absent | absent |
| TimeInfo | 5.345 | 16.993 | 16.319 |
| TickRegime | 4.575 | absent | absent |
| BookClockFlow | 1.915 | 14.344 | 14.364 |
| FlowResponseSurprise | excluded from B | absent | 5.812 |

C15/C18 concentrate approximately two thirds of model importance in session
range/time and anchors, with roughly another 14% in local BookClockFlow. B uses
several complementary roles that those fields cannot express on their own:

- `MaturedPathRegime` records fully elapsed sticky-path peak, path length,
  retention and peak dispersion; `MaturedBarrierOutcome` records matured
  bid/ask barrier outcomes, excursions and their elapsed duration. A cumulative
  session range and one 60s variation statistic do not determine these
  realized-history distributions.
- `TickRegime` includes one-tick price fractions and legal-band/boundary context.
  A common five-tick target is physically comparable but can have different
  percentage/economic difficulty across stocks. Physical normalization alone
  does not imply equal conditional movement propensity.
- `CauseConditionedQueueResponse` and `LiquidityTransmissionState` separate
  observed execution/queue-response effectiveness, defense and exhaustion;
  touch-relative work and net supply alone do not determine whether the same
  work translates into retained displacement.
- `AuctionCarry` expresses opening/VI pressure, indicative drift and release
  context. Continuous-flow interpretation is incomplete for different native
  observation regimes; its existing fields are not proof that recurring auction
  cycle information is already represented or available everywhere.

The live source contracts supporting those role distinctions are under
`src/oms/modules/feature/microstructure/event_path/{matured_path_regime,matured_barrier_outcome}/`,
`market_state/regime/tick_regime/`,
`microstructure/queue/response_lifecycle/cause_conditioned_queue_response.*`,
`microstructure/liquidity/liquidity_transmission_state/`, and
`market_state/exchange/auction_carry/`. These are semantic neighbor checks;
the immutable original native values, not a rebuilt current source, produced
the scored B/C arrays.

This makes loss of complementary magnitude/context information a credible
explanation. It does not identify which omitted family causes the loss:
importance can move among correlated fields, raw scale can proxy symbol or
liquidity, and three arm fits cannot isolate every omitted role. Restoring the
largest-importance families by default would be an outcome-guided expansion,
not a tested new information hypothesis.

The Flow fields were not ignored: C18 allocates 3.852% to response coupling,
1.213% to flow surprise and .747% to response innovation, total 5.812%. Their
use without verified gain is compatible with redundancy, unstable conditional
references, availability information or loss of conditioning interactions.
This comparison does not separate those explanations. `known_trade_share`
has zero model importance in both compact fits; that is model usage in this
experiment, not proof that attribution quality never matters.

## Consequences for the next information hypothesis

The strongest next source-grounded question is H16's **observed actual-auction
cycle ledger**: past accepted actual matches establish price/mass anchors;
subsequent simulated clearing proposals, residual-book basis and previously
realized proposal error may encode movement opportunity and direction when
continuous signed-flow attribution is unavailable. This changes the information
clock and representation rather than extending a short-window feature. Existing
AuctionCarry already owns several primitive inputs, so novelty must be the
recurring accepted-match/proposal relation and its causal publication; it must
not be claimed from its name or B importance. `H16_PLAN.md` independently fixes
that proposal and source semantics. Native source support is required before
predictive comparison, and no failed H15 mapping gate is rescued.

A separate falsifiable representation question is **conditional work-to-price
translation history**: does a durable completed episode ledger of successful,
contained and failed displacement, conditioned on the actual observation regime
and legal price-grid difficulty, carry magnitude information that current work
or cumulative session range misses? Existing matured outcomes, queue-response
and liquidity-transmission modules are direct neighbors. A future proposal must
first show exactly which causal conditioning/join they do not already publish;
otherwise it should be rejected statically. This is not a request to append all
of those existing families, lengthen their windows or perform a Cartesian scan.

For either question, the failed C15/C18 replacement is evidence against assuming
that this compact context is sufficient. Any new coherent replacement must
declare magnitude-opportunity, directional and observation-state roles before
labels, retain exact native missing/zero distinctions, and face a fresh baseline
under one frozen judge. Event support, TRAIN importance and native integrity are
prerequisites or diagnostics, never predictive admission. V5 supplies a resolved
negative comparison and more precise next questions; it supplies no accepted
representation or final OOS proof.

## Evidence references

All paths below are relative to `AstraResearch/`:

- `runs/information_state_20260930/v5/{frozen-contract.yaml,frozen-contract.yaml.identity,preflight.yaml,completed.yaml}`.
- `runs/information_state_20260930/v5/nomination.yaml` and `development-decisions.yaml`.
- `runs/information_state_20260930/v5/c18-versus-c15.yaml`.
- `runs/information_state_20260930/v5/fits/{baseline,c15,c18}-300-{calibration,development}-metrics.yaml`.
- `runs/information_state_20260930/v5/fits/{baseline,c15,c18}-300.yaml` and the corresponding saved CBM models.
- `experiments/information_state_20260930/{V5_PLAN.md,H15_COMPACT_C_SOURCE_AUDIT.md,H16_PLAN.md}`.
