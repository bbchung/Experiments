# H16 raw observation routing, prospective version 3

This is the implementation contract before materializing a new module or
reading its output. H16 V1/V2 source, binary, configs, methods and outputs remain
immutable. This version fixes observation routing, not a predictive failure or
a feature-selection result. No FE scoring has occurred for H16.

## Evidenced routing failure

V2 producer `763d32db7070f12b02656f79eb0e08ec89fa21af1baf17a924311773e5670194`
passed the unchanged four-file H15 registration byte control and completed both
all16 daily native jobs. Frozen support method
`81e9538c0d733593dd8c7e685f8e0d642f11f6541c8086381a13e58830f85000`
rejected origin-mid parity. Parity-only forensic comparison, without Alpha or
label reads, found 1277 changed 10s mids for Jan19/2409 and773 for Jan19/3037;
all original keys match in all32 cells. New mids become NaN where original mids
equal17.25/301, exactly their bound daily upper limits. Other30 cells match.

TwseFilter normalizes continuous zero-price touches to contract limits and may
merge equal-limit queues; removing it changes the original CurrentBook MID.
It separately rejects Trade quantity<=0 or price<=0, including legal simulated
zero-match snapshots. New routing must retain both the original MID contract
and raw simulated observations. Neither guard may be weakened or relaxed.

## New native material leaf

Type `AuctionCycleRawInformation`, centralized under
`src/oms/modules/feature/experimental/auction_cycle_raw_information/`.
Symbol is required, exactly one ordinary-share Book/Trade contract; positive
IndicativeMaxAge defaults10s. Seven Alpha names, physical tick/5 math, capacity
share, source-cycle-age and category meanings remain the H16 information
hypothesis. This is not a Python feature or an admission of its predictive power.

Keep original TwseFilter (RequireTradable false), CurrentBook, raw MD carrier
order and original PeriodicSampler unchanged in the new producer configs.
TwseFilter's original normalizing/filtering actions must remain authoritative
for the original MID/keys. A CRIT private tap before its HIGH filters copies
the accepted incoming header and complete raw Book/Trade into one bounded
source-ordered queue. It always returns true; it never rewrites a message,
advances Clock, emits an event, updates Alpha/Info, replaces msg_hook or publishes
a synthetic Quote. Module callbacks drain the captured raw payload only, never
reprocess the normalized Quote as another source fact.

Each raw row gets a private ordinal. Feed seq is not a raw identity because
filtered messages reuse it. Normal post-clock callbacks drain only through the
corresponding current raw ordinal, after matching source type/header. The
producer configuration uses direct native Feed publication with no interception
hook. Unsupported callback ownership/reordering makes state unknown.
The CRIT Clock listener drains only raw receiveR<Clock.now; equality stays private
until a genuine post-clock source callback. This preserves pre-message origins
exactly atR. A dropped zero-quantity Trade or observerless empty Book is consumed
at the first actual later native publication timeA; never backdate A toR.
No EOF, timer or expected auction deadline creates an execution or closes a
batch. Clock processing of a captured source row is not an inferred source row.
Real receive-prefix proof must use a genuine carrier D>S before the next target
source message, with comparison confined to original common origins<=D.

The queue has capacity256 and bounded first/latest overflow header markers.
Earlier queued rows must remain valid for origins before the lost interval.
Lost rows are never interpreted as price/quantity observations; hard-censor that
episode at its real availableA, preserve source order, and permit rearming only
from later complete valid source facts. Expose an exact cumulative overflow
counter. Day reset has separate private staging and published-state epochs.
ReceiveR and own exchangeE are independently guarded positive integers<2**53;
there is no E<=R/A/S assumption.

## Source time and public availability

R means the actual native incoming QuoteHeader receive time, including the
loader's existing monotone-receive contract. E remains reported own exchange
time. A is the Clock.now of actual raw-payload consumption/publication. All
public operands have A<=originS; safe sourceR<=A. Neither axis is substituted
for the other. Source freshness uses S-R; interval T is the positive difference
of first sourceR of two already observed distinct actualE groups. Received age
is (S-R_actual_first)/T, not a future deadline or modulo. Repeated actual prints
do not reset R, A, T or mass; unique same-E prints update only AS-OF mass/A.

Historical acceptance error requires matched prior Trial Trade/Book ownE and
both strict sourceR<actual_first_sourceR, plus fresh source times. These source
facts must all have been consumed before error publication; delayed co-consumption
does not claim an earlier public proposal availability. Same-R terminal
attribution remains unavailable. Book-only newer trialE invalidates an old
current proposal; absence cannot become known zero. Positive-volume legal
actual prints are immediate accepted facts; zero TRIAL is a simulated snapshot.
Residual basis uses raw paired positive-quantity legal touches and independent
freshness. Continuous raw zero-price market touches are valid partial input and
produce typed inactive numeric+0; only the unchanged original TwseFilter path
supplies canonical MID reference. Malformed raw data remains a hard boundary.

Retain the22 H16 Info names: first/previous anchor receive clocks and interval
are sourceR; fields named available_time now explicitly mean actual A. Add five
non-Alpha Info fields: actual_anchor_available_time, trial_receive_time,
residual_book_receive_time, boundary_receive_time,
cumulative_raw_overflow_count. All clocks/counters/known quantities are exact
nonnegative integers<2**53; absent operands preserve explicit missing states.
No raw price/quantity/time/counter enters the model. Alpha normalization stays
producer-owned physical tick/5 or same-symbol capacity/observed-cycle ratios.

## Required validation before any FE comparison

Focused tests must install real TwseFilter and CurrentBook: prove a continuous
zero-touch packet leaves original MID/keys unchanged, while the private raw
operand retains zero-price identity. Prove dropped zero-quantity/price0 TRIAL is
known-zero mass after genuine post-clock consumption, absent Trade stays unknown,
normal raw rows process exactly once, same-R origins see previous state,
R1<S<R2 staging cannot overwriteR1, and queue overflow fails closed. Retain the
actual identity/mass, pairing, independentE offsets, price-grid/sign/zero,
suspend-while-HALT, stale-view and receive-day-reset boundary tests.

After actual Release/focused/full test proof, freeze a fresh producer capsule
and helper/profile before commands. Reuse immutable runtime/input artifacts,
never old invalid H16 material outputs. Retain Jan19/Jan20/all16 original cohort
and the same six predeclared batch source cells/minima (5 session cycles,
5 sampled cycles,20 supported30s origins). Freeze the evaluator before reading
native values. Require ALL32 original key/mid bits, fieldwise state/sourceR/A
and integer semantics, zero observed queue overflow, and real prefix parity.
Field variability is descriptive per hypothesis; constant historical error
alone does not reject all other fields. These checks establish source integrity
only. All future CatBoost comparisons require a separately frozen common judge,
fresh baseline and numeric-ledger versus regime-only ablation.
