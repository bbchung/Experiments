# H16: observed auction cycle information

Prospective native implementation and support-study contract. The user has
authorized material C++ FeatureModule research; root controls builds, freezes,
fixed TRAIN pilot replays and label-free integrity/support checks. This document
does not admit a predictive representation or authorize a model comparison
without a separately frozen FE evaluation contract. The fixed16-symbol H15
mapping FAIL remains unchanged. H16 keeps every original
symbol/day/origin, including continuous and unknown regimes; it does not repair
H15 by deleting unsupported auction stocks.

The hypothesis is that a recurring call auction has an information clock defined
by already observed actual matches. Trial clearing price/quantity between matches are
conditional clearing proposals, while the latest actual match supplies a realized
price and observed matched-mass anchor. A cycle ledger of proposals, realized acceptance errors
and current proposal mass may contain direction and magnitude information over
60–300s. Extending continuous order-flow windows does not express that ledger.

## Source evidence and existing coverage

The TRAIN projection's metadata contains40 AuctionCarry and4
TwseStabilizationReference predictor names, plus10 PriceMemoryState names. This
inventory reads names only, not availability, values or predictive performance:
`runs/fe_origin_20260930/study-stock/train/projection.json`.

AuctionCarry already stores trial price/quantity, visible pressure and actual
auction trade data; H16 must not claim those primitive inputs are new.
[AuctionCarry.cpp](../../../src/oms/modules/feature/market_state/exchange/auction_carry/auction_carry.cpp)
lines160–165 select the opening event until `continuous_seen_`; lines450–486
publish `write_values` only on a continuous Book. The auction branch updates
internal state and returns. Its auction Trade handler, lines489–501, also only
updates state, and its header lines65–68 has no clock publication hook. These
source paths do not publish a repeatedly resetting actual-match cycle during an
all-day periodic auction. Existing opening/VI carry is not an admission of this
new representation. This is a source prediction, not a measured all-day missing
value census of the original baseline.

[TwseStabilizationReference.cpp](../../../src/oms/modules/feature/market_state/exchange/twse_stabilization_reference/twse_stabilization_reference.cpp)
lines128–175 requires `continuous_seen_` for a known reference and rejects a
noncontinuous book for its distances. Auction prints before a continuous phase
are held as pending opening price (236–246). It does not maintain the proposed
recurring accepted-match/proposal ledger. PriceMemoryState discards any nonzero
status Trade (36–41) and publishes from continuous books (107–125).

[TwseParser.cpp](../../../src/marketdata/parsers/twse_parser.cpp) lines245–269
preserve independent reported exchangeE and receiveR, map the wire trial bit to
TRIAL and the collective matching bit to AUCTION, and assign the same status to
Trade/Book. Lines261–321 emit Trade before the resulting Book. The quote-rule side
uses a prior continuous book only (271–283); UNKNOWN side on an auction print is
expected and cannot veto an actual match or create a directional trade sign.
The native parser test `AuctionPacketsPreservePerMessageVolumeInsteadOfReplacingWithCumulativeTotal`
at `src/marketdata/test/twse_parser_test.cpp:348–369` distinguishes per-message
actual volume from cumulative total.

The [TWSE ordinary-session disclosure description](https://www.twse.com.tw/zh/announcement/disclosure-report-d.html)
states that securities with extended matching intervals receive simulated
clearing price/quantity and five-level quotes every5s, with possible2-minute
matching delays. The [2022 introduction notice](https://www.twse.com.tw/staticFiles/news/news/tsecnews/ff8080818262fb8a01830ce92846024f.pdf)
predates the2026 TRAIN dates. These are documentary mechanism evidence; the
specific daily regime cannot be inferred from today's generic regulation page or
absence from an incomplete metadata snapshot. H16 does not consume an unadopted
notice classifier: its recurrence scale uses only already observed actual-match
receive intervals, and its category describes the current raw observation.

The [B.12.00 format description](https://www.twse.com.tw/staticFiles/product/broker/ff80808166388ea1016676ac941001e0.pdf)
pp31–33 separates trial disclosure, matching method, per-message volume and
cumulative volume. Five-level quotes are residual orders after matching or trial
matching, not the full demand/supply curves. H16 therefore does not reconstruct
a clearing curve or signed executed pressure from those residual queues. The
historical PDF is supporting field documentation, not proof of the exact capture
version of a particular BIN file.

The subsequently completed, independently frozen Jan19 label-free source census
is `runs/information_state_20260930/native-h16-support-census/census.yaml`, method
`8ef8bd2c54957b748206635e913b42a5bbb56ec20470551deb44968bfc09d727`.
It reports48 positive AUCTION/nonTRIAL/nonSUSPEND Trades for each3481/2344/2337,
with one print in each of48 distinct own-E groups; positive TRIAL|AUCTION Trades
number2831/2829/2792. The2330 control has one actual opening-auction print and6458
continuous positive prints. This confirms the existence of actual anchors and
simulated proposals on that one day; it does not validate quantity completeness,
trial price variability, predictive power or a two-day support gate. Observed
consecutive exchange intervals are slightly longer than300s, so exact deadline
or modulo resets are not justified. No new tape was decoded for this document.

## Falsifiable information claim

1. **Actual-cycle identity:** a last actual accepted match and a new trial cycle
   remain observable even when no continuous status occurs. This falsifies the
   premise that a continuous resumption is necessary for usable auction context.
2. **Conditional price proposal:** a trial price's change relative to the actual
   anchor, and disagreement between the current residual-book mid and trial
   clearing price, may distinguish future mid movement from a repeated level
   already reflected in today's mid.
   Last-cycle actual-minus-terminal-trial error tests proposal reliability rather
   than treating a trial price as an execution. Directional failure is plausible:
   the current residual-book mid can already contain all proposal information.
3. **Capacity and phase:** trial matched quantity relative to last actual matched
   quantity, conditional on observed cycle age, may discriminate magnitude even
   when direction is uncertain. A large proposal mass is not itself a future
   large move, and auction age does not establish that a match occurred or will
   occur at an exact deadline. Neither claim is accepted by source support alone.

## Native observation and missing-state contract

An eventual material implementation must be one new C++ FeatureModule under the
central experimental feature folder, exported through ValueWriter. Python may
join exact keys and evaluate receipts; it may not translate the feature formulas.
No frozen H15/H13/H10 or AuctionCarry producer is edited for this proposal.

An **actual match candidate** is a legal positive-price, positive-volume Trade
with AUCTION set, TRIAL and SUSPEND clear, and safe positive integerE/R<2**53.
TRIAL always has precedence as simulated state. Bare AUCTION Book is not an
execution; continuous Trade is not an auction anchor. The actual price and
per-message executed quantity are observed facts available immediately at their
native receiveR. A distinct later actual ownE identifies a new actual anchor;
same-E retransmissions cannot create extra cycles or reset age. If multiple real
prints share thatE, the reference size is their exact AS-OF observed quantity
sum, not a complete batch total. Sum in unsigned wide integer arithmetic before
division; repeated `(E,physical price,per-message quantity,cumulative quantity)`
identities do not add mass. Cumulative quantity is an identity/order guard only;
no cumulative-delta execution inference or trade signing is performed. A repeated
cumulative count with conflicting identity, unrecognized backwards cumulative
count within a group, conflicting clearing prices
within one purported auction group, unsafe clocks, own-E/receive regression,
suspend or invalid physical price make that current episode unknown.

No EOF or clock tick manufactures a zero-volume actual match, resets an anchor,
seals a group, or retrospectively publishes availability. Receive and exchange
regressions are separate guards. There is no E<=R, E<=A or E<=origin assumption.

The terminal proposal for an actual anchor must have been observed at strict
R_trial<R_actual_first; same-R ambiguity is missing, never a pre-match proposal.
Each raw trial Trade supplies an observed simulated price/quantity at its ownR;
it is not executed flow, and its price/quantity do not require a future group
closure to become observable. A last actual anchor is a historical fact; current
proposal freshness is a separate received-data state. For the five-second disclosure mechanism,
current proposals older than10s in received time are stale until a new
legal proposal arrives. Clock publication updates age/staleness only. An observed
pause/hard boundary invalidates current proposals and cycle attribution; it does
not relabel a previous simulated print as actual or invent a continuous epoch.

TwseFilter's Trade filter rejects quantity<=0 (`twse_filter.cpp:56–59`). A zero
quantity TRIAL record can be a valid simulated no-match snapshot. The H16 native
config therefore observes the original raw Book/Trade streams without TwseFilter,
with explicit producer-owned semantic guards. It never infers zero from an
absent filtered record. This routing and identical sampler/raw-mid control are
frozen before a pilot; existing baseline feature arrays/bytes stay immutable.

## Compact representation: at most seven Alpha outputs

Let C be the latest observed actual auction clearing price, V_C the exact AS-OF
sum of unique positive actual-print quantities in that actualE, I the latest observed
trial price, M the latest legal two-sided residual-book mid, V_I the nonnegative
trial matched quantity, and D_prev the prior actual clearing price minus its
strictly prior terminal proposal. Let A_C be the first received availability of
that distinct actualE. T is the positive receive-first interval between the last
two already observed distinct actualE anchors. No next match or future interval
is used. All fields are native observations available<=originS.

| Output | Meaning and cross-symbol scale | Responsibility |
| --- | --- | --- |
| `proposal_displacement_5ticks` | exact full-ladder tick distance C→I divided by5 | direction and magnitude |
| `residual_mid_clearing_basis_5ticks` | exact full-ladder tick distance I→M divided by5; current legal two-sided residual Book only | direction and magnitude |
| `previous_acceptance_error_5ticks` | exact full-ladder tick distance terminal_trial→prior_actual divided by5 | directional reliability/context |
| `proposal_match_mass_share` | V_I/(V_I+V_C), wide integer denominator guard; same instrument/unit; AS-OF observed execution mass, not complete batch capacity | magnitude/capacity |
| `received_cycle_age` | (S−A_C)/T; strictly past observed receive interval; no modulo, countdown guarantee or inferred match | magnitude/timing context |
| `observation_regime` categorical | observed continuous / indicative / actual auction disclosure / unknown / suspended | whole-cohort observation mechanism |
| `cycle_state` categorical | warmup / inactive / anchor_only / trial_zero / trial_up / trial_down / trial_flat / trial_na / stale / unknown | exact source support and relation |

No price, tick size, raw quantity, trading frequency or LOB thickness becomes a
cross-symbol predictor. Five ticks is the scientific move unit of the unchanged
label, not a z-score or a volatility estimate. The exact stock ladder is applied
across bands after physical price/grid identification; branches use integer
physical equality/sign before converting to a floating output. A zero integer
numerator produces positive0.0. Denominators are positive by source facts, with
no epsilon or pseudo-count. Cycle overdue is an integer received-age>=T diagnostic,
not proof of delayed matching; reported age is allowed to exceed1.

For M, both residual touch prices must be finite, positive, physically legal and
ordered; missing sides or market-price0 leave only that basis unknown. No supply
curve, hidden quantity or market-order limit-price substitution is manufactured.
If a zero-quantity TRIAL record has price0, its quantity is known zero while its
price is unknown: mass share can be zero and price differences remain NaN.

Numeric fields whose operands are not known remain NaN after warmup. Before any
relevant observation they remain−inf; current observed continuous regime is
known inactive with positive zeros plus its categorical state. Partial support
is represented field by field: missing last-cycle error cannot erase a known
current proposal/anchor. Fewer than two past actual anchors leave only the age
field unknown; no notice completeness assumption or symbol deletion is needed.
A current zero trial quantity is known zero, with `trial_zero` taking precedence over price direction.
Unobserved or invalid quantity is not `trial_zero`.

## Implementation contract

This section is the prospective implementation contract, subject to independent
native tests and the frozen pilot gate before any predictive admission.
Native Type is `AuctionCycleInformation`; the new leaf is
`src/oms/modules/feature/experimental/auction_cycle_information/` with its own
header, implementation and CMakeLists. Export one slot per named Alpha family:
`AuctionCycleInformation.0.FIELD.0`, exactly the seven table names above. Numeric
registration is P2/S1 for the three price differences, P2/S0 for mass share and
received age; both categories are P2/S2 categorical. The cycle category's
`trial_up`/`trial_down` mirror relationship is declared producer-side; other
category literals are direction-invariant. No Python feature computation exists.

Typed parameters are `Symbol` (required nonempty string, matching the group's
single ordinary-share Book/Trade contract) and `IndicativeMaxAge` (positive duration,
default10s). The pilot uses only that default, with no expansion or parameter
search. `REQUIRED_STATUS_MASK` includes TRIAL|AUCTION|SUSPEND. A direct raw
Book/Trade subscription uses explicit accept-all `StatusFilter: 'TRIAL || !TRIAL'`
and no TwseFilter dependency. Fail module startup if it receives multiple symbols,
missing Book/Trade legs, or a nonstandard stock tick ladder. Use ModuleInit,
FeatureModule-owned feature families, normal callbacks, TIME_API clock and
retained subscriptions through keep_subscription; no shared framework edits.

Raw clock guards are positive integer microseconds<2**53, receive monotonicity
across both own Book/Trade callbacks, and reported exchange monotonicity within
that symbol's received observation sequence. These are separate axes. Invalid
clock/domain or malformed physical price is a hard episode reset, with all
affected context unknown until new source facts arrive. New day clears state;
previous valid history is never exported as current after a hard boundary.
Ordinary continuous observations instead publish all five numeric fields as
known inactive +0.0, clear current trial views and retain already observed actual
history for a later observed indication. A TRIAL transition is expected input,
not a hard reset or execution. SUSPEND takes precedence over TRIAL.

On the first unique positive actual auction Trade in a later actualE, capture
the latest fresh trial price at strict prior receiveR for the acceptance error,
then immediately update C, A_C and actual quantity. Reset current trial views;
new proposals belonging to this cycle require receiveR>A_C. Equality leaves
cycle attribution unknown, without deleting the observed actual price fact.
Update T only from the difference of the first receiveR of this and the previous
distinct actualE, never from later fragments, repeats, zero prints or a timer.
Within that same actualE, enforce one physical clearing price and aggregate only
unique observed positive print quantities. Store at most256 exact native
identities in that group; capacity overflow is a hard unknown-state boundary.
Unsigned128 arithmetic precedes denominator conversion; observable quantity
diagnostics must remain exact below2**53. A duplicate does not change mass, A_C,
T or the anchor counter. Unknown cumulative accounting is not imputed execution.

TRIAL quantity is a snapshot, never summed across observations. For nonnegative
quantity and legal price, replace I and V_I at that observedR. Quantity0/price0
is valid known-zero mass with unknown price; it must neither advance the actual
anchor nor censor it. Quantity0/legal positive price may retain that observed
trial price. Negative quantity, nonfinite malformed price or positive-quantity
off-grid/zero price is unknown hard input. Price-only and quantity-only operand
availability is preserved. The previous acceptance error is the latest actual
minus its fresh strictly prior received trial, not a future realization attached
to an earlier origin. No direction comes from aggressor side.

Residual M is the latest raw two-sided legal ordered touch midpoint with
strictly positive disclosed quantity on both best touches, with its
own received age<=IndicativeMaxAge. Its mid is identified on the exact physical
half-cent grid before full-ladder conversion; it is not an average of rounded
ticks. Missing sides, a disclosed empty book or market-price0 makes that basis
unavailable without deleting legal actual/trial facts. Malformed depth/order,
negative quantities or nonzero off-grid prices are hard invalid inputs. TRIAL
Book is allowed and does not establish a new actual anchor or a full supply
curve. The residual basis requires the latest residual Book and current TRIAL
Trade to share their own exchange timestamp. Trade-before-Book publication
therefore retains proposal displacement and mass as partial facts, with the
residual basis unknown until its matching Book arrives. A later Book-only
TRIAL exchange group invalidates the older current proposal without inventing
zero quantity. Terminal acceptance error requires a fresh prior TRIAL Trade
and its matching Book, both received strictly before the actual anchor's first
receive time; same-receive disclosure cannot provide terminal attribution.
Stale trial affects its dependent fields, while historical error and
observed age remain independently known. No epsilon changes any boundary.

Exact regime literals are `warmup`, `continuous`, `indicative`, `auction`,
`unknown`, `suspended`, describing the last received raw native status. They are
not all-day classification or a reusable eligibility rule. Cycle state precedence
is hard/suspended→`unknown`, continuous→`inactive`, no actual anchor→`warmup`,
current trial stale→`stale`, no post-anchor current trial→`anchor_only`, known
zero trial quantity→`trial_zero`, unknown price→`trial_na`, then exact physical
I−C sign→`trial_up`/`trial_down`/`trial_flat`. A warmup cycle category does not
erase a separately observable pre-anchor residual-mid/clearing basis.

Non-Alpha diagnostics, one slot each, are fixed before any pilot count read:
`actual_anchor_exchange_time`, `actual_anchor_first_receive_time`,
`actual_mass_available_time`, `previous_anchor_first_receive_time`,
`observed_match_interval_us`, `trial_exchange_time`, `trial_available_time`,
`residual_book_exchange_time`, `residual_book_available_time`,
`previous_error_available_time`, `cumulative_actual_anchor_count`,
`session_actual_anchor_count`, `cumulative_unique_actual_print_count`,
`cumulative_actual_repeat_count`, `cumulative_trial_count`,
`cumulative_hard_censor_count`, `actual_clearing_price`, `trial_clearing_price`,
`observed_actual_quantity`, `trial_matched_quantity`, `cycle_support`,
`boundary_available_time`.
Clocks/counters/source quantities are exact nonnegative integers<2**53 where
published as float64; an absent trial quantity operand remains NaN, while a
known trial zero quantity is exact positive zero. Unavailable price diagnostics
are NaN, never fake zero. Historical trial diagnostics may remain observable
after a newer Book-only TRIAL group makes that proposal unavailable as current
information; their presence does not override the native current-state predicate.
Session counter credits distinct actual first-R in[09:10,13:00); other counters
are day cumulative. `cycle_support` is an exact source predicate: observed
actual anchor plus fresh strict post-anchor TRIAL observation, with known
nonnegative quantity or known legal price; it does not require any outcome,
nonzero feature or all five numeric operands. Export these as Info/metadata,
not AlphaFactors, and do not use raw diagnostics as model inputs.

Framework halt ordering must be tested explicitly: Contract::update_trading_halt_state
uses TRIAL or SUSPEND to enter HALT (contract.h74–83), and on_trading_halt has no
reason parameter. An expected TRIAL halt callback must preserve the actual
anchor. Empty Book publication advances the clock and updates contract status
without a Book observer (quote.cpp505–512); SUSPEND while already TRIAL/HALT
may produce no halt callback. The producer must observe the accepted raw
boundary reason without publishing Alpha or emitting events from a msg_filter.
The agreed strategy is a private, always-accepting IDLE raw tap which stages only
headerR/E/status and an empty-Book boundary flag in source order. It never stages/uses future
TRIAL price or quantity for feature publication. No feed msg_hook is replaced.
Normal callbacks consume a pending actual boundary only after native clock
advancement; an exact current raw TRIAL-only reason makes on_trading_halt
non-destructive, while SUSPEND/unknown halt reason is hard. An empty received
Book clears the residual-book view without deleting unrelated legal price facts;
an actual empty SUSPEND clears the entire episode even if HALT was already set.
If no observer callback runs, only the first Clock.now>rawR may apply that staged
real boundary. It records `boundary_available_time=that actual publication time`,
never backdates it. At a pre-message origin exactly rawR, all Alpha/Info stay in
their prior state. A staged hard boundary cannot be overwritten by a later
same-R expected TRIAL before it is consumed. The private boundary queue is
bounded at256 entries and retains earlier unconsumed boundaries. Overflow
retains the first dropped header as a forced-hard marker at its own received
time, with bounded last-dropped chronology/status if needed; no dropped price
or quantity is reconstructed. Its attribution is unknown until new legal
source facts arrive. Multiple empty disclosures must apply in accepted source
order rather than allowing a latest pending slot to erase an earlier boundary.
This consumes already received
source evidence; it never manufactures a match, expiry outcome or EOF closure.
A private staged receive-day identity resets only staged R/E comparisons before
checking the first incoming header of a new received day. It does not publish
Alpha state or erase that header. Public day advancement later resets the
ledger/day counters. Own E monotonicity is enforced within a received day;
an old-day E leading the new-day E cannot by itself reject the new day.
Unit cases must cover both transitions, same-R
source ownership, partial price/quantity, all zero/missing states, repeated and
fragmented actual prints, cross-band mid math, both independentE offsets and
real native clock-before-dispatch origin order. Clock ticks must never create
an execution, cycle identity, quantity or acceptance error.

## First empirical gate: fixed unlabelled TRAIN source study

Freeze this support procedure before any new census/replay. Fixed dates are
20260119 and20260120, fixed original16 symbols and original PeriodicSampler/key
cohort. The fixed recurring-auction source cases are3481,2344,2337, identified
before this proposal's feature-value study by the independent native source
census; all other symbols remain in the export and comparison as current
observation-regime controls. No date replacement, symbol substitution, notice
lookup or eligibility filter is permitted. Missing required data gives incomplete evidence, not a new
day or silently smaller successful cohort.

The cheapest preliminary measurement is a root-only, single-pass census of the
already bound Jan19 uncompressed16-stream prefix-selection spools: status mask,
Trade positivity, distinctE, per-message/cumulative quantity, and R/E cadence.
No new compressed tape is decoded for this census, and no labels are touched.
It validates whether actual AUCTION/nonTRIAL positive execution groups and valid
TRIAL price/quantity actually coexist. A negative result rejects the proposed
anchor or routing semantics before native implementation; it cannot be cured by
signing auction prints against a continuous book.

The frozen Jan19 source census has now established this narrow anchor premise.
Root may freeze and run a new native pilot on the
same two fixed TRAIN dates. Required integrity: every existing original key and
raw-mid bit is preserved in every present cell; original rows are not filtered;
all new numeric/category/state/clock contracts pass; repeat snapshots preserve
cycle identity/first-receive age/mass; same-E fragmentation with identical final
observed total preserves AS-OF mass without claiming complete batch coverage;
zero-trial quantity stays distinct
from missing; independently shifted positiveE preserves Alpha information; real
receive-prefix common origins preserve every field. No EOF/timer proof is allowed.

Required source support in EACH of the six fixed auction symbol/day cells:
at least5 distinct actual anchors, at least5 distinct anchor/proposal cycles
represented on original30s origins, and at least20 such origins. Report finite,
known-zero, unknown and underlying integer/physical operand variation for each
numeric hypothesis separately. Report positive/negative/flat price relations and
zero/positive trial mass without turning one sign's absence or a zero acceptance
error into automatic rejection of every other source hypothesis. These are
observable source/semantic-support checks, not tail-label or predictive
thresholds. The5-cycle/20-origin minima retain existing pilot-scale controls,
without tuning them to this candidate. A degenerate field can be dropped only in
an explicit new representation version after the frozen source-support result;
that version must be frozen before FE scoring. It is not a comparison outcome
rescue, model tuning or automatic variant search.

Report every current observation-regime state for all other cells without demanding
auction cycles there. This is a typed inactive regime, not H15's continuous
execution contract. Preserve the original all-regime mid-price endpoint labels
unchanged. Actual auction-price prediction can be a later distinct scientific
target, but cannot substitute for the requested mid-price label.

Only after native support and observation provenance pass should root freeze a
fresh FE comparison with identical baseline/candidate sampling, labels, splits,
CatBoost and judge. Separate the categorical mechanism context from the full
ledger in a bounded ablation so a gain cannot be attributed to new actual-match
information merely because a regime category is useful. Report pooled and
predeclared positive-batch/continuous/unknown groups, and horizon decay at
60/120/180/300s under one frozen evaluation procedure. Group diagnostics do not
select symbols. No change of judge or conditional-head method is adopted here.
