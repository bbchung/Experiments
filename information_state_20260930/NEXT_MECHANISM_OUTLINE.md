# Next mechanism outline: acceptance of quote-led price marks

Status: independent reasoning proposal only. No producer, predictor, labels,
sampling, universe or training procedure has been implemented or changed for
this proposal. This document is outside the V3 scientific source closure.
H10 results must be interpreted before choosing the next actual study.

The next information source worth testing is whether a quote-led price mark
subsequently acquires execution evidence on both sides of its new spread.
This is a hypothesis about observable information arrival, not proof of
informed trading, news, hidden liquidity or a participant's inventory.

The motivation comes from the earlier failures. H9 encodes observed sampled
mid renewals, dwell/clock priors and signed runs, accepting only physical mids
and origin keys (representation_research_v2/renewal.py:1-6,21-33,186-220).
Its V2 300s development comparison was unsupported: baseline versus renewal
joint AP was 0.127601 versus 0.127332, big AP 0.264158 versus 0.258923, and
pooled conditional direction AUC 0.585522 versus 0.589159. Equal-symbol-day
direction AUC declined from 0.557745 to 0.535733. No candidate was nominated.
These exposed development observations motivate a different information
source; they do not authorize changing gates or selecting favorable stocks.
They also caution against equating a small pooled directional gain with
broad cross-symbol information.

H10 asks about known demand, subsequent quote-only repair, touch renewal and
later same-side demand. The proposed next mechanism instead begins with a
new quote price mark and asks how the market tests that proposed level. A
provisional mark, execution on only one side, and later known execution on
both sides can have the same current mid while carrying different evidence
about whether that level is accepted or likely to be reversed. The explicit
prediction hypothesis is that mark acceptance changes signed continuation
versus reversal at 60-300s. Any magnitude claim must be separately declared
and validated; persistence of a level need not imply another five-tick move.

The first source audit found important neighboring representations, which
must be controls rather than ignored. PriceFormationPath stores signed flow,
price-path summaries and cause categories (price_formation_path.cpp:24-100),
using log normalized quantity/depth primitives (:438-445). FlowResponseSurprise
compares interval flow and movement with a prior local linear response
(flow_response_surprise.cpp:107-112,132-162). PressureResponseState's support
state follows displayed quantity beyond the old price anchor and its observed
coverage (pressure_response_state.cpp:339-376). Those inspected formulas do
not join subsequent bilateral executions to the identity of a particular
quote mark. A scoped source search found no explicit bilateral acceptance
state. This is not a completed proof over every registered producer: the
remaining family/alias inventory must be checked before implementing a new
family, and equivalent existing state should be reused if found.

The proposed representation is a small episode rather than another rolling
return or activity window. Establish a strictly observable quote mark with
legal current bid/ask. Keep its price identity and receive availability.
Classify later executions only after their exchange clusters close, using
the known side and legal contemporaneous book. Track whether the new bid and
ask have both been tested in later distinct clusters; revoke or censor the
state when the level leaves the observable domain or a halt/gap/ambiguous
cluster invalidates it. Known zero, warmup and unobservable states remain
distinct. An execution at a historical visit to the same price must not be
silently credited to the current visit. Closed-state publication uses actual
availability, with a real receive-prefix proof before predictive scoring.

The price level owns the state lifetime. A confirmed level is historical
information until it is explicitly revoked or superseded; it is not made
more informative by simply choosing a longer lookback. A clock-age cap may
protect data freshness, but its semantics must be distinguished from that
level's acceptance. TRAIN-only native inspection must determine whether
this distinction survives the current 30s scoring origins. A lack of sampled
support is a rejection of this representation/sampling recipe, not a reason
to invent replacement days or secretly move to event sampling.

Cross-symbol scales should express the episode's market meaning: known work
at each side divided by the strictly positive displayed capacity recorded
when that side of the mark was established; signed displacement in units of
the five-tick target; and shared categorical acceptance states. A zero or
unknown reference is an uncomputable state, not a denominator epsilon.
Raw quantity logs, raw quote counts and raw arrival rates should not enter
the new family. The static baseline audit already identified six unscaled
PriceMemoryState log-volume Alphas (price_memory_state.cpp:14-17,97-103);
logging raw volume is not a stock normalization.

The full baseline remains an unadmitted control. A later compact replacement
must retain target-scaled movement capacity, session/price-anchor context,
quantity/depth demand, supply response and causal cadence/volume references.
It should distinguish absolute capacity to traverse five ticks from relative
activity surprise; normalizing all volatility away discards event base-rate
information. This compact context is a separate preregistered representation
problem, not permission to normalize 3254 columns mechanically or to accept
the failed 27-column core. Any activity/tick-regime eligibility or grouping
must use independent TRAIN-only rules and held-symbol safeguards.

Before any fit, use one bounded representative TRAIN native pilot to test
identifiable provisional and bilateral-confirmed episodes, both directions,
observable price domains, correct missing states and actual sampled origins.
Then compare against the neighboring state above. Counterexample streams
should show the new episode retains mark/visit information that the proposed
marginal control drops. If it is recoverable from an equivalent existing
state, do not add another family. No label-based search across all features,
horizons and symbols is proposed.

Only after support and source novelty pass should a comparison contract be
frozen. A small baseline/complement/marginal-control comparison can test the
episode claim under one common label, population, split, training and judge.
If H10 already supplies equivalent information, first test a compact
replacement or a direct ablation; adding both is not a default. Directional
evaluation must include equal-symbol-day, held-group and paired-day evidence,
not merely pooled AUC. Magnitude power must remain in the complete retained
representation. Future sealed OOS remains required for actual admission.

There is no mathematical incompatibility between magnitude and conditional
direction in the ideal three-class cross-entropy objective. Write
b=P(up)+P(down), q=P(up|big). For a neutral label the loss is -log(1-b); for
a big label it is -log(b) plus the corresponding binary direction loss.
Thus the loss decomposes into binary magnitude cross-entropy plus direction
cross-entropy only on big outcomes. A finite shared CatBoost tree budget and
global early stopping may underweight the rarer directional task; that is
an empirical training hypothesis, not an inherent impossibility or proof
that neutral dominance caused the observed FE failures.

A separate baseline-only study could hold one CatBoost magnitude head fixed
and fit a CatBoost direction head on native big labels, scoring every original
origin. No alternative learner, HPO or FE changes are necessary. Freeze the
original-population versus conditional-population weighting choice explicitly,
along with disjoint early-stop/calibration roles and head persistence. Retain
direct b and q in the score contract: reconstructing b from the two composed
joint probabilities can create ULP changes in ranking ties. Validate any
claim-specific judge separately before applying it to another FE comparison;
the current V3 judge remains unchanged.

The legacy astra.signal hybrid is not that protocol. signal/model.py:62-79
fits a LogisticRegression calibrator on early-stopping big labels using a
clipped CatBoost regression and the classifier's odds, with epsilon clipping
and float32 composed scores. It returns only the MultiClass model, so adopting
it as a reproducible predictor would also require freezing/persisting the
regressor and calibrator. Reusing ES as supervised calibration is a different
role contract; it is not automatically future leakage, but cannot be imported
into the frozen V1/V2 comparison. No legacy source has been edited here.
