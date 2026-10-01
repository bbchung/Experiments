# H15 independent trade-led external-information proposal

Status: NEW prospective plan, not frozen or executed. This supersedes only the
unexecuted H15 quote-led proposal. H13's admission, sampling and support gates
remain exact; its failure cannot be rescued by this mechanism. No actual market
data, counts, labels, model or score files were read for this plan. No native
implementation, replay, preprocessing or training is authorized by this file.

## Information-source hypothesis

A known one-sided peer execution episode and an observed same-sign peer mid
displacement can mark external price discovery. The target can follow partly,
absorb opposing flow, or continue moving later. Does this source-confirmed
history, together with the target's subsequent response, improve the 60--300s
five-tick problem relative to a compact local representation?

The hypothesis includes continuation and absorption AFTER an initial target
response. It is not limited to unresolved subsecond latency. Receipt ownership
establishes what this engine could know; it does not identify exchange-clock
leadership, informed participants, economic causation or a fundamental value.

Peer admission requires positive known one-sided work against an observable
prior book AND nonzero same-sign physical mid displacement. One touch moving
is sufficient. Confidence-one exact-touch and through-touch executions qualify.
There is no zero-print formation, both-touch renewal, second-side execution,
minimum shock ratio, move-size cutoff, work threshold or searched latency.
Physical nonzero and source validity supply the only strength admission rule.

## C++ ownership and source boundaries

Proposed native leaf is
`src/oms/modules/feature/experimental/peer_trade_information/`, with a typed
`PeerTradeInformation` FeatureModule. One target instance owns its target and
single reference leg. `TargetSymbol` and `PeerSymbol` are explicit; no dynamic
best-peer choice occurs. Both Book/Trade streams follow the corresponding
native `TwseFilter` route with `RequireTradable:false`, plus required status/halt
delivery. Trading permission is not a pure-signal eligibility criterion.

`Quote<T>::contract()` identifies the leg, header receive time/sequence identifies
receipt availability, and Book/Trade supplies the leg's exchange clock.
`Module::bind_feature/bind_input` orders native feature dependencies;
`follow_quotes` follows an existing producer's feeds. A typed value accessor by
itself establishes neither subscription nor causal ordering. Prefer direct
two-leg ownership for this first implementation. Keep per-leg book references,
`CausalTradeSideInference`, pending clusters and source epoch state separate.

`CausalTradeSideInference` assigns confidence one to price>=fresh ask or
price<=fresh bid when feed direction is consistent; exact equality is not needed
for directional execution evidence. Caller-owned status, ordinary-share price
grid, book observability and freshness remain mandatory. Feed-side/location
conflicts, inside-spread/tick-rule fallbacks, unknown/stale inference and mixed
side clusters cannot be promoted into known execution.

The implementation must aggregate execution quantities by their actual prior
book/capacity identity. A print is inferred against a causally previous received
book; a later bounding book must not retroactively infer its side. All observed
prices use the supported ordinary TWSE-share ladder, exact canonical physical
keys and legal positive queues. Mid displacement is arithmetic raw-price mid
on that ladder, not a sticky-price label or mean quote-tick proxy.

## Two clocks, closed clusters, and mark lifecycle

Each leg independently has monotone effective receive R and monotone exchange
E. Validate exact integer clock boundaries, positive values below 2^53, own
source epoch and own-book freshness of at most five seconds on each axis.
Never compare peer E with target E or with S/R as an ordering rule. E may lead
R. Independent constant +/-one-hour E changes must preserve every predictor.

An own-leg E group commits only at the receive time A of its next distinct-E
callback. Positive unknown/conflicting or mixed prints invalidate the entire
group's execution attribution. Nonpositive prints follow native filter semantics
and do not close it. EOF and logical-clock ticks do not close pending groups.
The fixed 256-callback buffer rejects an overflowed group as a whole; it cannot
publish an earlier partial mark. These are integrity limits, not HPO parameters.

On a clean closed peer execution group, freeze its first valid pre-execution
book/mid and the work of each same-side prior-capacity identity. A same-sign
nonzero peer mid displacement observed after that execution can occur in the
same ordered group or a later valid closed book group. A book that preceded the
execution cannot satisfy the required subsequent displacement. Before a valid
displacement, retain one pending execution episode; ambiguity, a hard boundary
or a book-observation gap that invalidates its reference ends that attribution.
This five-second observation validity is not the completed mark's lifetime.

At the first qualifying displacement, freeze one historical source mark:
unique ID, direction s, pre-execution peer mid, observed displaced mid, normalized
closed work, and actual closure availability A. Further turnover does not
recount or strengthen that completed mark. A subsequent clean peer execution
episode can create a new mark and supersede ownership. Current return through
the old source anchor records refutation; it never rewrites past confirmation.

The target baseline is its latest valid CLOSED book with availability R<A and
receive age<=5s at A. No future target book can backfill it. A target callback
at equal R=A supplies a tied category and cannot establish peer-first credit
from loader order. Later target execution/book facts belong to this mark only
when their actual receipt R>A; own pending clusters still commit on closure.
Every update must be available by its sample S. All16 carrier streams and source
dispatch order remain fixed, preserving genuine periodic origins.

Retain at most one latest confirmed peer mark and one target response prefix.
Target pair changes and initial partial following do not terminate the history.
No 60/300s TTL exists. Age is a predictor. Current target displacement requires
a fresh valid target book; current peer persistence separately requires a fresh
valid peer book. Stale observations publish missing views without destroying a
historical fact. Unknown/mixed current target demand loses that work attribution,
not the completed peer fact. Hard status/book-grid/clock/source-epoch boundaries
clear affected attribution; a broken target baseline cannot bridge a new epoch.

## Seven numeric and two categorical fields at most

For each closed, known execution-capacity group j define
`wj=Xj/max(Qpre_j,Xj)`, with Xj>0 and Qpre_j>0. Aggregate exact wide quantities
BEFORE division within one capacity identity; splitting a print must not create
extra work. Wp is the confirmed peer work. Wbuy/Wsell are known target work
since A. This describes executed work against observed capacity, not gross
liquidity, order identity or realized trading cost.

Let mT be the frozen target mid. Let u_s(mT)>0 be the absolute log-price unit of
a physical five-tick move from mT in peer direction s on the COMPLETE target
share ladder. It supplies a target-event-relative percentage scale, not a beta
forecast or expected target destination. Do not fit a coefficient on later TRAIN
days and backdate it into this pilot.

| Numeric field | Meaning |
| --- | --- |
| `external_impulse_target_unit` | peer observed signed log-mid displacement / u_s(mT): external percentage impulse in the target event's unit |
| `peer_work_fraction` | Wp/(1+Wp): dimensionless closed execution pressure; not a raw volume or gross-depth fraction |
| `target_response_5ticks` | exact current target physical mid displacement from mT / 5 |
| `target_known_work_imbalance` | (Wbuy-Wsell)/(Wbuy+Wsell), explicit known-zero convention versus unknown attribution |
| `target_known_work_strength` | log1p(Wbuy+Wsell), observed accumulated capacity-relative work |
| `source_persistence` | current peer RAW half-cent mid-key displacement from its pre-execution anchor / its SIGNED, exact NONZERO frozen raw confirmation displacement; a dimensionless fraction of the source economic-price impulse, not a tick-distance ratio |
| `mark_receive_age_300s` | (S-A)/300 seconds, with no expiry at one |

Numeric history is shared bit for bit between the controls below. Source and
target raw price/quantity/depth/trade-frequency magnitudes are not predictors.
Quantity scaling preserves work fractions. Full ladder target units preserve
five-tick base-rate information rather than z-scoring away absolute volatility.
Analytical log units are not rounded into invented quote identities. Positive
queue and nonzero exact source displacement guards precede division; no epsilon
or inverse repair can turn an unknown/zero state into a finite signal. Initial
warmup is -inf; post-warmup missing required references are NaN. Observed no work
is exact zero, distinctly categorized from ambiguity. Unsupported symbol/ladder
groups fail closed. Comparable units do not prove uniform pooled sensitivities.

J has two categories: `receive_relation` (waiting, strictly later quote, strictly
later flow, tied, invalid) and `response_alignment` (unresponsive, following,
opposing, absorbed, source refuted, unknown). Define absorption strictly as
ANY exact positive known target work with no observed same-direction target mid
progress; it is an observed conditional fact, not hidden-liquidity proof. Exact
integer work presence controls this category, never the sign/zero of a floating
buy-minus-sell ratio. Precedence is refuted, unknown, following, absorbed,
opposing, then unresponsive. Freeze literals within native categorical capacity.

## Fair limited control and source-support gate

K exports the SAME seven numeric J values, references, availability, sentinels,
history and row keys. Only its two categories differ: independent peer process
phase and independent target process phase. Both are causal, known at the same
S, and derived from the same primitive facts. Export J/K in one native feed pass.
Do not duplicate numeric arrays or rerun identical preprocessing.

This contrast isolates the added chronological/alignment CATEGORY CODING at
equal numeric and categorical capacity. Its numeric history is already joined
at A, so it is NOT a complete causal-join removal or proof of economic transfer.
C+K versus local C asks whether this external joined history helps; C+J versus
C+K asks whether explicit chronological categories add information/usable
representation. State both inferential limits in any results. No future-value,
day/symbol shuffle, artificial delay or altered reset interval is permitted.

BEFORE counts, freeze fixed Jan19/20 jobs and both target2308/peer2317 and
target2317/peer2308. Same full warm-in, all16 clock carriers, import/raw routes,
BasicInfo, original 10s 09:10--13:00 periodic rows and existing30s subset; no
Labelers or invented event samples. Freeze source/binary/config/input receipts.
Native all-book outputs are diagnostic snapshots at book receive, not every
execution/cluster callback and not a replacement evaluation cohort.

Gate for each available fixed day/target cell: >=5 qualified marks with valid
target baseline and >=5 distinct RECENT mark IDs observed at fresh existing30s
origins, with 0<=S-A<=300s. Overall: >=5 qualified marks in each direction and
>=20 distinct existing30s origins with semantic ORDERED RESPONSE SUPPORT:
valid historical mark, fresh source and target views, and a target CLOSED quote
or known execution whose receipt R>A, plus at least one nontrivial relation:
nonzero target displacement; ANY exact positive known target work without same-direction
target progress; or physical source refutation while that strictly later target
evidence is present. Emit a native diagnostic predicate for this definition.
J/K category string inequality is not support: their different vocabularies
would make that count tautological. Report K phases descriptively, without an
isolated causal-transfer claim. Pre-first-response and post-response occupancy
are descriptive separate counts; neither is mandatory. Recent300 is a support
measurement scope, not state TTL. Report missing cells, ties, stale/unknown/mixed,
overflow, hard boundaries and pre-session mark carry-in separately. Session
completion counters cannot credit pre09:10 warm-in facts as session completions.

Exact parent five keys/raw float64 mid bits, all Alpha/sentinel states, A<=S,
old/new control invariance, per-leg E-offset tests and genuine receive-prefix
nonflush proof must pass. Support and integrity are not predictive admission.
A failed gate rejects this exact mechanism/sampling combination. No dates,
symbols, gate thresholds, source admission, windows or ages are relaxed after
counts. A new hypothesis would require its own new prospective contract.

## Prospective universe mapping and later representation comparison

The pair pilot tests native primitives/support only. It must not select a best
peer or lead to two populated columns surrounded by missing H15 state in the
original sixteen-symbol pooled universe.

One FIXED prospective full-universe mapping is designated-reference mapping:
every original target uses peer2330, except self2330 uses peer2317. These are
declared native reference roles, not externally verified sector leaders or an
empirical best-beta claim. The existing fixed target list and held-label rule
remain unchanged. There is no contemporaneous TRAIN-wide covariance fit,
calendar backdating, target-specific HPO or calibration/development selection.
The pair pilot and this mapping are explicitly different scopes; success on
2308<->2317 does not prove coverage or predictive value of the full mapping.

Before any full material or FE comparison, independently freeze a cheap
unlabelled native coverage/integrity gate for THIS mapping on fixed TRAIN cells,
then reject or retain the mapping on those prospective rules. Do not replace
failing references after counts. An alternative restricted two-symbol universe
would be a separately frozen universe/method question before new baseline
comparison, not a fallback success claim. No market taxonomy was looked up or
assumed in defining these deterministic roles.

Support pilot does not require rebuilding C20. Eventual compact local C must
first audit actual native reusable outputs/zero semantics and choose a bounded
coherent role set: target movement scale/geometry, local direction and demand,
session/price anchors, causal activity surprise, and observed supply response.
The old27 core is not automatically retained. Prefer existing dimensionless
Alpha outputs and already comparable full-ladder physical tick outputs. A constant
division by five is not required to admit an existing physical-tick unit. Native
price log returns have percentage meaning; this differs from H15's new
target-five-tick-relative impulse unit. Selecting frozen native columns in Python
is orchestration, not feature math. Do not force a new session/VWAP adapter or
simultaneously rewrite five native state machines just to start this experiment.
Declare omitted roles before labels and freeze the SAME C input artifact for C,
C+K and C+J. Audit exact zero/epsilon/+1 shrinkage semantics first: raw log-volume
is not normalized by logging, and absolute volume shrinkage constants need an
explicit scale interpretation.

At most four primary CatBoost fits: fresh qualified full control B, compact C,
C+K, C+J. One independently validated/frozen judge applies identically to all
arms, with original pure-mid labels/sampling/splits/training/metrics/acceptance.
Any formulation change must first complete its separate method study and then
restart at a fresh baseline. Predeclare the C+K/C and J/K contrasts and their
direction/magnitude claims. No extra complement arm, secondary fit grid, HPO or
best-horizon nomination is planned. Future sealed OOS remains the final proof.

## Overlap and exact inspected source anchors

`CrossLeadLagResidual::align_time_return/publish` aligns same-window returns and
has mapping/freshness, not known execution-to-later-target ownership.
`CrossReturnContext` is held-mid same-window return/volatility context.
H6's sampled covariance graph does not recover these event-cadence execution
marks. `LiquidityShockLifecycle` and `PostSweepOutcome` are local attack/shock
neighbors; applying their summaries to a peer is not by itself H15 novelty.
External source confirmation, receive-qualified target history and explicit
chronological coding are the proposed differences, with the K limitation above.

Proposal source anchors are `src/oms/modules/module.{h,cpp}`,
`feature/order_flow/trade_flow/trade_side_inference.h`,
`feature/experimental/quote_visit_acceptance/quote_visit_acceptance.{h,cpp}`,
`feature/microstructure/cross_symbol/{cross_lead_lag_residual,cross_return_context}/`,
`feature/microstructure/liquidity/liquidity_shock_lifecycle/liquidity_shock_lifecycle.cpp`,
`feature/microstructure/execution/post_sweep_outcome/post_sweep_outcome.cpp`,
`feature/observable_book.h`, and
`writer/dataset_writer/dataset_writer.cpp`, all beneath `src/oms/modules/`.
These live inspected sources are not a frozen compiled producer receipt.
