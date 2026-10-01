# H10: oriented demand, delayed repair and renewed demand

Status: hypothesis and TRAIN-only support research. No predictive comparison,
model fit, native replay or AlphaFactor admission has been performed for H10.
V1 and V2 scientific sources/profiles remain unchanged.

## Research question and source boundaries

If V2's price-renewal representation fails, test whether future large endpoints
depend on demand that survives supply repair and quote-level changes. A buyer's
first attack can be absorbed, displayed supply can rebuild at later book-only
events, and the buyer can resume at a renewed price. The ordered, side-preserving
sequence can differ from temporary imbalance with the same total volume,
depletion, refill and past price movement. This is an observable proxy for
persistent demand, not identification of a parent order, hidden liquidity,
individual cancellation or trader identity.

Generic depletion, persistence and refill are already extensively represented.
The parent's TRAIN projection lists PressureResponseState (37 numeric fields),
LiquidityTransmissionState (95), TradeClusterPersistence (12),
SignedTradeClusterState (40), TradeRunRefillDeficit (1), QueueRaceState (12),
QueueDepletionRun (36), QueueResilienceCurve (12), LiquidityVacuumContinuation
(60) and ReleaseTiming (33). Their presence rules out claiming that another
rolling pressure/refill aggregate is a new information source.

Exact live-source boundaries:

| Native module | Existing information | Boundary relevant to H10 |
| --- | --- | --- |
| PressureResponseState | Same-side runs with interruptions; original-price refill/absorption; later quote-only supply is already accumulated | `pressure_response_state.cpp:196` resets runs on 5s idle, 30s age or sufficient opposite work. Unknown/mixed timestamp trades reset the run. `update_levels` at line 301 accumulates supply on **every book**, including attack=0. Only `last_response`/`response_change` update on attack>0. An original-price level survives a touch change while observable; touch renewal itself is **not** a reset. Age/idle/gap/resumptions are registered but excluded by the parent's AlphaFactor-only projection. |
| AbsorptionReleaseLifecycle | Same-touch attack and cumulative repair, including quote-only repair | `absorption_release_lifecycle.cpp:269` completes an episode at touch-price change or its 5s deadline. Outcomes omit side and both sides enter the same Context; active outputs take the maximum across sides. Ordered cross-level, side-preserving continuation is lost. |
| LiquidityTransmissionState | Directional attack/survival/reload/defense sums, repeated attack price and current counterfactual ladder shape | `liquidity_transmission_state.cpp:195` computes reload inside the loop over pending executions. A later interval containing only quote repair does not attach a repair event to the earlier attack. Window totals and last attacked price do not retain the proposed chronology. |
| TradeRunRefillDeficit | First book response after a qualifying completed same-side run | `trade_run_refill_deficit.cpp:39` clears pending runs after that book and pools directions. Its constant finite value does not establish absence of later repair. |
| QueueResilienceCurve | Original-price recovery at 100ms/500ms/1s, with censoring | `queue_resilience_curve.h:31` and `.cpp:101` use a 5s recovery episode; this is already a recovery path, not evidence that repair is absent. |
| SignedTradeClusterState / TradeClusterPersistence | Event run shares, regularity, signed wall-time pressure and 30s/120s decay | They already include 240s/480s or 300s history. H10 must test conditional chronology across repair/renewal, not merely longer memory. |
| LiquidityShockLifecycle | UP/DOWN shock episode across price changes; attacked/support frozen prices, repeated shocks and sticky displacement | `liquidity_shock_lifecycle.cpp:91` already preserves side and levels, but the actual lifecycle is 5s. It does not distinguish immediate versus delayed repair and join the first repaired mark to later new-touch execution. |
| QueueOutcomeSequence | Same-side, success-after-failure and failure-after-success ordered terminal queue outcomes | `response_lifecycle/queue_outcome_sequence.cpp` already expresses order and side. Its terminal tuple lacks exact execution price/quantity and prefix repair marks; underlying cause=EXECUTION uses aggressive/removed≥0.8, rather than strict exact-price execution matching. |
| QueueResponseLifecycleState / PostSweepOutcome | Terminal queue recovery; side-preserving sweep, original attacked/support refill and aligned markout | `response_lifecycle/queue_response_lifecycle.cpp:75` ends on touch change, recovery or 5s. `execution/post_sweep_outcome.cpp:177` retains actual sweep side and old prices, but does not condition delayed repair on later same-side demand at a new touch. |

Side preservation and event order alone therefore are **not** novel. The exact
new hypothesis is strict known execution → subsequently observed quote-only
repair → repaired mark retained through quote renewal → later same-side known
execution at the new touch. Net displayed replenishment is observable; gross
order additions, individual cancellations and hidden liquidity are not proven.

These live formulas explain possible representation limitations, not proof of
the exact frozen engine's values or a predictive gain. Baseline field inventory
comes from `runs/fe_origin_20260930/study-stock/train/projection.json`.

## Existing TRAIN evidence and exploratory pilot

`train/health.csv` finite values are constant for TradeRunRefillDeficit (1),
PressureResponseState absorption bid/ask last_response and response_change (0),
and run_known_share (1). Their availability states still vary. This is explicitly
allowed by the `nonconstant` calculation in `study_data.py:48`; the screening
flag is not proof of nonconstant finite values. Cumulative supply has variable
values, so immediate response=0 must not be interpreted as no delayed repair.
Run quantity/reference ratios reach 18,208 and refill recovery reaches 2,393;
one initially tiny queue is a fragile scale for a pooled demand episode.

An exploratory, conservative raw census used **20260119/20260120 × 2337/2609**,
the first two declared TRAIN dates and two fixed symbols. It read 1,194,685
compressed bytes, ran in approximately 0.51s on one CPU thread, read no labels
and fitted no model. Script, raw/contract/source hashes and primitive definitions
are retained under `runs/information_state_20260930/h10-pilot/census.py` and
`receipt.yaml`. The final exchange-time cluster is right-censored; a cluster's
availability is the receive time of the next distinct exchange-time row.

| Cell | Strict known quantity share | Containment → later price renewal → later same-side touch demand |
| --- | --- | --- |
| 20260119 / 2337 | Undefined: no continuous trades | 0 |
| 20260120 / 2337 | Undefined: no continuous trades | 0 |
| 20260119 / 2609 | 0.97934479 | 66, including 30 adverse renewals |
| 20260120 / 2609 | 0.97954507 | 54, including 28 adverse renewals |

The proposed numerical support gate arrived **after these exploratory counts**.
This run is not retrospectively preregistered and does not pass a support screen.
Both 2337 cells contain only raw StatusMask=3 in the 09:10–13:00 scope, so a
four-cell gate requiring at least five chains per cell necessarily fails.
Symbols/dates are not replaced to rescue it. This prototype did not require
delayed repair or the next pilot's 30s demand-gap rule.

For 2609, instantaneous `current old-touch qty - previous old-touch qty + known
executed qty` was never positive across 2,289/2,626 contained touch challenges.
This can reflect the tape's immediate execution/book ordering; it neither rules
out quote-only repair at later observations nor proves a C++ defect.

## Exact raw-state, label and export meanings

`src/msg/md_msg.h:28` defines TRIAL=1, AUCTION=2, SUSPEND=4; raw StatusMask is
parsed in base 16 by `TradeBookMd::CsvLoader`. Mask 3 therefore means native
TRIAL|AUCTION. It does **not** identify a public disposition, batch-auction cause,
actual execution or exchange schedule by itself. BasicInfo records day_trade=No
for 2337 and Yes for 2609 in both pilot dates; this is a separate trading-permission
field and is not a pure-signal mechanism eligibility criterion.

The inspected native endpoint configuration explicitly sets
`TwseFilter.0.RequireTradable: false`
(`native-endpoint-stock-release-1/2337/config.yaml:123`). TwseFilter passes
noncontinuous books unchanged; for continuous books it maps a zero-price touch
to the daily limit and merges a duplicated next limit level. Disposition flags
are imported only through the separate Disposition input, not inferred from
raw StatusMask or BasicInfo day_trade.

Native mid validity differs from continuous-mechanism validity.
`Quote<Book5>::update` (`quote.cpp:415`) computes `(bid+ask)/2` when both touches
have positive price and quantity, independently of status bits. CurrentBook has
no configured StatusFilter in the inspected config; the default accepts all
statuses (`quote.h:40`). `ReturnLabelerBase::ReturnLabel::value`
(`return_labeler.cpp:24`) checks Y availability, not TRIAL/AUCTION status.
Its Y is `CurrentBook.0.book_mid_ticks.0`; noncomputable origin/endpoint Y produces
NaN rather than a negative class. Thus a finite noncontinuous mid can have a
pure-price label while continuous order-flow modules are unavailable. The raw
census has not established native DatasetWriter row presence for either 2337
cell; do not equate raw rows with the parent sample cohort.

Config-only pilot export is possible with `DatasetWriter`:

- Remove PeriodicSampler, BookFlipFilter and BookFlipSampler to capture each
  accepted book (`dataset_writer.cpp:703`); no Labelers are necessary.
- Set EmitSampleContext=true, UseTmp=true and a fresh dedicated OutputPath.
- Use exact named MetadataExports for CurrentBook raw mid/bid/ask ticks and
  PressureResponseState age/idle/gap/resumptions. Named metadata bypasses the
  AlphaFactor filter (`dataset_writer.cpp:438`). Bound value dependencies order
  producer updates before the writer.
- Existing public values do not expose exact side-specific lifecycle IDs,
  completion events and reference-price identities. CurrentBook also does not
  expose the full raw price/quantity ladders or every trade's side/sequence.
  BookStructure has log raw touch/side-depth metadata, but its inverse is not a
  substitute for raw integer quantities in exact event attribution.

A precise native pilot therefore needs a narrow diagnostic-only sink or optional
metadata: per-side episode ID, start time, anchored price/reference quantity,
completed outcome ID/time/kind, attack quantity, later quote-only matched-price
net supply, censor/reset reason and source receive/exchange/book sequence. Log
at existing update/completion branches without altering published FE math. Keep
book and trade event order; native trade-side inference is a proxy, not proof of
actual aggressor or hidden parent identity. No such C++ instrumentation is done
in this research note.

## New pilot preregistration: eligibility and fixed support scope

The following next pilot is separate from the failed exploratory prototype.
It must be written and hashed before its eligibility counts or selected-symbol
support counts. No endpoints, model scores, calibration or development outcomes
may enter selection.

1. The universe is the exact sixteen symbols in V1's frozen profile. Eligibility
   uses the **first twenty declared TRAIN calendar slots**, 20260119 through
   20260224, not twenty observed replacement days. Missing raw/BasicInfo files
   are empty exposure and are reported; the calendar never extends.
2. Stream raw rows once, on one CPU, without frames, a row cache or full dataset
   duplication. Scope only receive-clock 09:10 inclusive to 13:00 exclusive,
   applying CsvLoader's monotone receive-clock clamp. Count raw book status and
   native two-touch mid validity separately. No public auction/disposition label
   is inferred from the bits.
3. A usable continuous book has no native TRIAL/AUCTION/SUSPEND bit, both finite
   positive touches/quantities and positive spread after source-equivalent zero
   touch normalization; depths lie in [1,5]. A symbol qualifies if at least 90%
   of its scoped reported books are usable continuous books. Zero book exposure
   is unsupported and does not qualify. Report all twenty availability slots
   and both status-only and usable-book fractions; no day_trade exclusion.
4. Select the first two qualified symbols in ascending symbol order. The support
   dates remain exactly **20260119 and 20260120**. Missing pilot cells are skipped
   and reported without replacement dates or symbols. Insufficient cells make
   the fixed support proof inconclusive, not an invented pass.
5. Resolve only positive known-side trades against a fresh **previous** strictly
   usable book: native CausalTradeSideInference confidence=1, with both receive
   and positive exchange-clock age at most 5s. No current/future quote inference,
   midpoint/tick-rule fallback or feed/location conflict becomes certain demand.
   Mixed-side same-exchange-time clusters and unknown executions censor the
   whole ordering. A cluster is closed only after a later distinct exchange-time
   row is received; the final cluster is censored.
6. A completed observable chain is: known trade at the prior opposing touch;
   that same touch remains positive at the following book (containment); a
   **later book-only interval** shows positive net displayed increase at that
   same observable price (delayed repair); a later distinct exchange cluster
   changes the opposing touch; and a still later known same-direction trade
   attacks the already renewed touch, observable in both bounding books. Never
   treat a price leaving a full L5 ladder as known zero or cancellation.
7. Fix MaxGap=30s between same-side known demand observations, exceeding the
   native local RunGap=5s to test survival across plausible order-splitting pauses.
   This is one semantic rule, not a gap sweep. Unknown/mixed events, noncontinuous
   status, invalid/stale books, backward exchange time, opposite demand and
   nonobservable tracked prices censor the chain. Completed earlier states are
   retained only as already observed information, never revised into success.
8. Support requires at least **five complete chains in every present fixed
   day×symbol cell**, at least twenty combined chains and both directions overall.
   Report zeros, censoring reasons, observation denominators and waiting times.
   This proves implementable observable support only; it cannot admit Alpha or
   claim prediction. If the screen fails, reject the proposed implementation
   scope before a full native FE replay or GPU fit.

Filesystem metadata was inspected without decompressing the eligibility scope:
320/320 raw csv.zst files exist; total **448,472,858 compressed bytes** (427.7MiB),
maximum file 7,573,203 bytes. Root subsequently authorized one streaming pass;
the eligibility results must be stored separately with this preregistration's
hash and raw/source/script provenance. The original four-file receipt remains
unchanged.

## TRAIN support results and exact-price correction

The preregistered eligibility scan completed once in **89.25 seconds**, using one
CPU and streaming 320 compressed files without row frames or a duplicated cache.
All sixteen symbols qualified. The deterministic ascending selection is **2308,
2317**, with usable-continuous fractions 0.994551 and 1.000000. All twenty calendar
slots and the four fixed January 19/20 pilot files exist. This selection used no
labels or FE/model scores. In particular, 2337's two exploratory noncontinuous
days do not establish its twenty-day regime: the fixed aggregate usable fraction
is 0.969935, versus native mid-valid fraction 0.973684. No public exchange event
classification follows from these counters.

The first implementation reported 78 repair→renewal→same-side-resumption chains.
Inspection of retained examples found that its latest renewed touch could return
to the original repaired anchor before resumption. That proves a broader renewal
path, but does not establish execution at a **touch distinct from the original
repaired price level**.
The original script and receipt are preserved without overwrite. A second,
explicitly recorded semantic correction requires the resumption touch to differ
from the original anchor under native price equality. Its implementation receipt
was written before rereading the same four small files and declares that earlier
counts were seen; it is not an independent unseen support experiment. No
eligibility, date, gap, known-side rule or support threshold changed.

The corrected exact-price sequence has **52** completed chains, 17 buy and 35
sell. Each fixed cell exceeds the original five-chain support requirement:

| TRAIN day | Symbol | Complete chains | Buy / sell | Mean / maximum elapsed seconds |
| --- | --- | --- | --- | --- |
| 20260119 | 2308 | 16 | 7 / 9 | 16.34 / 65.49 |
| 20260119 | 2317 | 15 | 5 / 10 | 5.07 / 13.49 |
| 20260120 | 2308 | 10 | 4 / 6 | 22.65 / 82.11 |
| 20260120 | 2317 | 11 | 1 / 10 | 7.00 / 31.42 |

The script excludes anchor-return resumptions and repair first observed after an
earlier touch renewal. Positive, negative and mixed-cluster boundary fixtures
passed before the first census; distinct-price and repair-before-renewal fixtures
passed before the correction. Stored examples satisfy strictly ordered exchange
events and nondecreasing observation availability. The gap cap is between known
demand observations, so total episode length can exceed 30 seconds. Longer
support lifetimes alone do not prove 60–300s predictability.

Ignored provenance is under
`runs/information_state_20260930/h10-pilot/preregistered-v1/`:
`preregistration.yaml`, `H10_NOTES.preregistered.md`, `eligibility-receipt.yaml`,
the unchanged broad `support-receipt.yaml`, and
`distinct-price-v2/{support-implementation,support-receipt}.yaml`. The original
preregistered notes snapshot remains immutable even though this live notes file
now includes results. Receipts bind the scripts, calendar, selection, raw inputs,
BasicInfo and native primitive source hashes. The preregistration SHA256 is
`a6e60d2e7aa9e2ae7ac8f89a5dc5a5efbefe073a34968cee2b9aea840f63e922`.
All seven bound native primitive sources and receipt identity links were checked.

This supports implementing an observable causal mechanism pilot. It is a
conservative independent raw-event proxy, **not native module parity, Alpha
admission, label predictability or final OOS proof**. There were no labels, GPU
fits, native replays, C++ changes or changes to the frozen V1/V2 comparison.

## Proposed compact representation and bounded comparison

Only proceed after source/causal support validation. Maintain a side-oriented
episode rather than a rolling sum. An episode links completed challenges and
repair/renewal states; counter-demand, ambiguity and observation breaks terminate
or censor it. All quantities are displayed-order-flow proxies.

The native mechanism implements eight fields with these fixed roles:

| Native field | Role | Semantic cross-symbol scale |
| --- | --- | --- |
| direction | Direction only | Known episode demand sign s in {-1,0,+1} |
| log_work_strength | Magnitude only | log1p(W), W=sum Xj/max(Qpre_j,Xj); every positive execution challenge contributes at most one queue-equivalent unit |
| signed_after_containment_share | Direction/persistence | s×W_after_containment/W, with supported positive integer challenge denominators |
| signed_after_repair_share | Direction/persistence | s×W_after_delayed_quote_only_repair/W; earlier repair is known before later execution |
| delayed_repair_fraction | Magnitude/context | R/(R+frozen first max(QA,XA)), where R is observed quote-only net replenishment; no hidden-liquidity identification |
| signed_renewed_work | Direction and magnitude | s×W_newprice_after_repair, using the same bounded per-challenge work scale |
| mid_progress_5ticks | Direction and magnitude | Exact physical share-ladder millitick coordinate displacement/5000; true zero and half-cent midpoint boundaries survive |
| phase | Process/observability context | Shared string category warmup/inactive/contained/repaired/renewed/resumed/stale/censored |

Work denominators use each challenge's positive displayed touch quantity and
executed quantity before division. Integer support and net repair decisions
precede floating conversion. This prevents a tiny queue inflating work while
retaining one-lot positive repair. Shares are bounded with defined support.
The supported calculator is the native TWSE share ladder: physical quote/trade
keys are validated and canonicalized within bounded ULP equivalence before
inference and state comparisons. Off-grid prices and unsupported ladders censor.
No universal z-score or numerical epsilon changes event identity, zero, ratio
validity or observation boundaries.

Current episode history is **UNBOUNDED**: MaxDemandGap30s bounds an idle gap,
not total episode age; freshness remains a separate MaxBookAge5s contract.
Numeric warmup is -inf, closed continuous inactive is exact zero, and censored
state is NaN until fresh legal containment. RESUMED retains the earlier repair
mark and counts its first completion only. Seven raw timestamp/price/counter
diagnostics are named metadata, excluded from predictors. Directional and
magnitude metrics remain separate under the existing judge. The shared phase
literal `censored` fits the native categorical storage contract; long reason
strings must not silently truncate into invented categories.

The global LogicalClock also expires the published view after 30s since the last
**closed** known demand: phase becomes `stale`, numeric values NaN. It does not
close an open exchange cluster or discard the internal episode; later genuinely
closed input can establish current information at its actual new availability.
An exact 30s age remains supported; 30s+1µs does not. A late closed cluster is
subject to the same current-availability gap check before publishing.

A future frozen comparison uses fresh full baseline B, B+H10 ordered state,
B+H10 with causal `ResetAtTouchRenewal: true` ablation, and optionally B+exchangeable
challenge counts/sums with no chronology. Maximum four primary CatBoost fits;
omit the last diagnostic for three. Preserve exactly the V1/V2 judge, sampling,
labels, cohort, splits, training formulation and acceptance rules. Internal FE
state ablation is the research variable. No universe changes from OOS results.
Full-baseline complements remain attribution controls; legacy raw normalization
must still be resolved before declaring the final complete representation valid.
The ordered/reset native pilot shares inputs, freshness, gap, origin keys and
scales with only that retained-mark difference; see
[NATIVE_H10_PLAN.md](NATIVE_H10_PLAN.md). Native replay has passed the four-cell
source-key, physical-mid, availability and support checks described below;
predictive validation remains pending.

If H10 fails, question the source and information-release regime: continuous
event flow versus native noncontinuous/indicative books can require different
observable state and clocks. The known-phase grouping rule is itself a next
representation hypothesis, not justification for retrospectively dropping
low-performing symbols or changing this round's judge.

## Native pilot evidence and historical producer boundary

The immutable H10 binary `8678aa7fd4cac798909211164284423b5e12c0e32ca021367a05962a5e07b0bd`
was replayed on the four fixed TRAIN cells. The separate ignored
`native-h10-ordered-reset/independent-native-validation.yaml` reports exact
original int64 key order, 1,381 rows per cell, float64 raw OriginMidPrice bit
agreement, finite current availability no later than the origin, valid phase and
numeric sentinel states, and 461 origins per cell on the frozen 30s grid. No
pilot label column or predictive metric was read.

| Day/symbol | Full-prefix native completed chains buy/sell | Resumed origins on 30s grid buy/sell |
| --- | --- | --- |
| 20260119 / 2308 | 22 (8/14) | 3 (1/2) |
| 20260119 / 2317 | 18 (8/10) | 1 (0/1) |
| 20260120 / 2308 | 12 (5/7) | 5 (3/2) |
| 20260120 / 2317 | 11 (1/10) | 2 (0/2) |

These satisfy the predeclared event/sampled support thresholds, conditional on
producer lineage. Full-prefix episode counters do not assert parity with the
resetting Python proxy's 52 completions. Root's final build also passed all 23
focused H10 cases and all 101 registered CTest cases.

The two 20260203 control replays are byte-identical between preserved current
producer `69b512...` and H10 producer `8678aa...`. They are **not** byte-identical
to the frozen historical producer `2374c82...`. The historical SOURCE_AUDIT.md
already documents the intentional FlowResponseSurprise NaN-to-warmup correction
and CrossReturnContext halt correction after the predictive freeze; those frozen
scientific inputs were preserved. The old native screen's historical gate thus
remains failed. Current/original keys, raw mid and all 16 native labels agree bit
for bit in both controls, independently of those feature-state differences.

Before any V3 FE comparison, the component-provenance method must independently
validate and freeze the original producer's reproduction, current H10 invariance,
exact accounted historical differences, and candidate original-key/mid binding.
This changes neither the FE judge nor historical baseline values. Pilot support
does not authorize full replay, predictive admission or an OOS claim.
