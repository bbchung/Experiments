# H13 prospective source-support study v2

Status: READY for independent root review and method/profile/input freeze.
No actual source stream, origin table or label has been read for this study;
no support counts, native replay, feature comparison or model fit has run.
This document and the new package are outside the frozen V1/V2/V3 closures.

## Versioned correction before any actual support counts

Support method v1 `ade482d1e200d72f179e0d51d8f1ff2599f01031660c4c23f519cbc75b4d1a93`
and its package/profile/plan/tests remain immutable. A read-only synthetic
audit found that an already received touch-pair departure followed by a return
inside the SAME still-open E cluster could resurrect the old active phase:
snapshot previously compared only the latest received pair with the stored
visit pair. Closure later correctly discarded that visit, but an intervening
origin could incorrectly count old one-sided credit in the contrast gate.
No actual market stream or source-support count was opened before discovering
this source-semantic blocker.

This isolated v2 package adds only a monotone received-departure view guard
keyed to the unique visit ID. A departure invalidates that visit's view even
if the latest pair returns before closure. Only a NEW closed admissible visit
can rearm it. Ordered pending group evidence remains available for legitimate
historical completion at its actual later closure, and an already completed
landmark remains historical information. No source rows, input identities,
sampling, normalizers, fields, thresholds or acceptance gates are changed.
The versioned method/profile/proxy schemas end in v2; no v1 alias is accepted.
The prospective profile differs only in version and plan/test source proof.

Four added synthetic counterexamples cover same-E leave/return before an
origin, preservation of completed history, valid pending completion before
departure published only at later closure, and rearming only at new closed
admission. The original37 semantic tests are retained in the new test module.

## Question and evidence limit

Does a quote-led change of both touch prices subsequently acquire exact,
known execution at both sides of that same quote visit, and does this
information survive the original 30s origins as a distinguishable historical
landmark? This is an observable information-arrival hypothesis. Bilateral
testing does not prove informed trading, consensus value, defended liquidity
or another five-tick future move.

Existing PriceFormationPath describes normalized interval work/path/cause;
PressureResponseState follows displayed support beyond an old anchor;
PriceMemoryState describes price/volume histograms. SyntheticMeasurement's
TouchToTradeDelay already owns individual side waiting flags and first-touch
trade delays. H13 must therefore establish joint price/visit ownership rather
than claim that first executions, turnover or price memory are new sources.
H9's unsupported renewal/dwell comparison motivates testing a different
information source, not extending its windows or rescuing it with sampling.

This implementation is a **label-free, target-only replay proxy**. Even a
passed support gate is neither native FE support proof nor predictive
evidence. Actual native multi-carrier receive-prefix and source-parity checks
are required before any FE comparison. The frozen V3 producer, sampling,
labels, CatBoost formulation, metric and judge are untouched.

## Fixed population and prospective inputs

The only cells are 20260119/2308, 20260119/2317, 20260120/2308,
20260120/2317, in that order. These are the already fixed TRAIN cells; no
OOS, label, model or future-date selection occurs. Missing raw, BasicInfo,
parent origins or required symbol metadata is skipped without replacement;
the remaining present cells still face every per-cell and global gate. A changed present artifact
rejects integrity instead of silently accepting another producer.

`h13-support-profile-v2.yaml` copies exact input paths/SHA256 identities from
the existing H10 preparation, full-preparation and component-lineage
receipts. It binds the actual `.bin.zst` source, day BasicInfo, original native
Parquet, and prepared `expected-origin-keys.parquet` for each fixed cell.
`expected_keys_sha256` in H10 hashes that Parquet artifact, not flattened
key bytes. The profile also binds existing calendar/receipts and selected
immutable compiled C++ source copies; its data-file hashes are copied from
those existing receipts, not recomputed by opening market data now.

Only `SampleTime`, `SampleBookTime`, `SampleBookSeq` are read from origin
Parquets during an authorized future run. Each existing native table must
have the bound 1381 int64, non-null complete key rows and exactly match its
prepared key artifact. Select existing rows with `SampleTime % 30000000 ==
0`; require 461 increasing 30s keys. No new grid, label maturity check, row
invention, feature availability filtering, replacement date or interpolation
is permitted. Day and symbol come from the bound partition identity. The
future formal FE study will reuse its already shared mature cohort.

## Inspected raw ABI and clock contract

The package decodes inspected native x86_64 little-endian layout:

| Message | Size | Source-owned fields |
| --- | ---: | --- |
| MsgHeader | 24 | int64 seq at 0, int64 receive at 8, type at 16 |
| Book5 | 152 | E at 0, bid/ask arrays at 8/48, quantities at 88/108, status at 128, symbol at 132, depths at 148/149 |
| Trade | 56 | E at 0, price at 8, quantity at 24, status at 32, symbol at 36, side at 52 |

Source contracts are `src/msg/md_msg.h`, `src/sdk/types/coco_type.h`,
`TradeBookMd::BinLoader`, TwseFilter, CausalTradeSideInference, the ordinary
TWSE share tick ladder, Feed::publish and LogicalClock. Padding is opaque.
Corrupt/truncated/unsupported messages or a payload symbol mismatch reject;
no source record is repaired or manufactured. Native source precedence is
`.bin.zst`, then `.bin`, then CSV; this study permits only its explicitly
bound `.bin.zst` and never falls back.

Per-loader callback receive time is `max(raw_receive, previous_receive)`;
raw receive and wire exchange are retained as separate annotations. Effective
E falls back to that receive only when wire E is nonpositive. The two domains
must each be ordered, but E need not be less than receive or SampleTime.
An E regression censors continuity and clears the open transport group.

Close an exchange cluster only when the next different observed E callback
arrives; all its information is published at that closing callback's receive
time. EOF does not close the final group. Snapshot at origin S **before** any
callback whose clamped receive R is equal to or greater than S, matching
native clock-forward-before-dispatch order. No timestamp is shifted to make
an equality pass. Target-only reconstruction omits all 16-carrier tie order,
so it cannot establish complete native source-phase or producer parity.

TwseFilter suppresses nonpositive trade quantities; these cannot close an
observed group or poison a zero-positive-print interval. A reported positive
quantity with invalid/ambiguous price still censors this conservative raw
proxy, including cases where native TwseFilter would suppress an invalid
price. This difference is declared rather than advertised as native parity.

## Price, inference and visit ownership

Ordinary-share quotes must lie on legal native bands .01/.05/.1/.5/1/5 TWD
at 10/50/100/500/1000 boundaries. Recover physical half-cent integer keys
only if one price ULP plus one scaling ULP bounds the residual, with total
uncertainty below 1/8 of a grid unit and scaled price below 2^53. Off-grid
prices are censored, never rounded into a valid quote using an arbitrary
epsilon. Validate ordered visible levels, depths, quantities and spread.
Normalize native zero-touch limit quotes using the bound BasicInfo; refuse
an int32 quantity overflow rather than pretend parity over overflowing native
normalization. Full-ladder displacement uses exact integer milliticks and
midpoint identity; it is never price divided by the current tick size.

Admission requires both touch prices change in the same direction between
valid continuous books, positive initial QB and QA, **zero positive reported
prints anywhere in the whole closed formation E cluster**, and zero positive
prints since the preceding valid book. This includes feed-side-unknown and
prints arriving after the new quote within that same E group. The absence of
observed prints does not prove that trades did not cause the quote change.

Freeze the shared touch pair, its unique visit ID, mark direction, QB/QA,
exchange identity and receive availability. Later positive confidence-1 SELL
exactly at its bid and BUY exactly at its ask in distinct later closed E
clusters count as bilateral testing. Inference uses only a previously seen
book; both receive and exchange freshness are independently <=5s. A feed
unknown exact-touch print can have native confidence1; an inside-spread,
conflicting-side, stale or unknown-location print cannot. Known through-touch
prints do not test the exact anchor. Mixed signs/feed sides poison current
visit attribution for the whole group. Ordinary side ambiguity and
bounding-book gaps over5s clear unfinished visit/reference-book credit,
but cannot falsify an already completed historical landmark. No unknown work
is assigned a side. Status TRIAL/AUCTION/SUSPEND, a hard invalid book,
exchange regression, quantity overflow or a new day break the source epoch
and clear both current credit and the retained landmark. Book-gap freshness
is a current attribution guard, not a completed-landmark TTL or a market
activity explanation.

Any touch-pair departure ends unfinished credits; revisiting the same price
cannot revive them. Until the closed group is available, a raw received pair
departure suppresses the old active-visit view; it does not invent a new
quote-led visit. At first bilateral completion freeze the historical
landmark and its exact work. Further turnover at that visit does not update
its strength or generate additional completions. A completed landmark is
retained until another completed landmark or a hard source-epoch censor;
touch departure by itself does not erase an already completed fact. It
claims historical testing, not surviving queues. A stale receive-time view
publishes missing values, without erasing that completed fact, a timer
closing a group or counting an unfinished episode. A valid fresh view can
show that same historical landmark again. Ordinary unknown prints cannot
restore unfinished credits or backfill a completion. A new cell starts
with no state carried across days.

## Compact diagnostic representation and missing states

Five proposed fields are emitted for source diagnostics, not registered or
admitted AlphaFactors:

1. `visit_phase`: warmup/inactive/provisional/bid_tested/ask_tested/bilateral/
   untestable/censored/stale, with received-pair and book-freshness guards.
   Untestable is ordinary current attribution failure; censored is a hard
   source-epoch break. Historical fields can remain finite in untestable.
2. `completed_mark_direction`: +/-1 direction of the last completed mark.
3. `accepted_anchor_offset_5ticks`: signed current-mid minus completed-mid
   in exact full-ladder milliticks divided by5000.
4. `bilateral_test_strength`: min(EB/(QB+EB), EA/(QA+EA)), frozen at first
   completion. This normalizes testing work by positive offered capacity;
   it is not an absolute-move forecast.
5. `completed_receive_age_300s`: observed age divided by300s, neither a TTL
   nor a rolling lookback.

Work is summed as bounded exact int64 integers before floating division;
references remain the strictly positive queues frozen at visit admission.
No epsilon, inverse-small-error or quantity z-score is used. Exact equal
canonical mids produce positive zero offset. Before the first valid closed
book numeric diagnostics are -inf warmup; after that no available historical
landmark or a hard censor/stale view gives NaN. Phase inactive is observed absence
of a qualifying current visit; absent anchor is never fabricated as zero.
These are explicit proxy diagnostics; a later native feature contract must
be separately reviewed before formal export.

## Preregistered support gates

Session completions have availability within the first and last original
origins. Distinct recent sampled landmarks must be such session completions,
observed at an unchanged original origin with `0 < S - available <= 300s`.
The 300s boundary is a support-age gate only and does not erase a landmark or
change samples. Repeating one landmark across many origins counts once.

All checks must pass:

- At least one of the four fixed cells is observed; no replacements. Missing
  cells and whether all four are observed are reported descriptively.
- Each present cell has at least5 unique completed landmarks.
- Each present cell has at least5 distinct completed landmarks observed by a recent
  original 30s origin.
- Across fixed cells, at least5 up and5 down completions.
- Across fixed cells, at least20 original origins in provisional, bid-tested
  or ask-tested phase.

Report all phase/censor/event counts and the open final group separately.
Failing support rejects this exact representation/sampling recipe; do not
extend episode lifetime, change gates, use event sampling, pick other symbols
or fetch/replace dates. Passing reports `native_support_proved:false`,
`predictive_evidence:false`, `labels_read:false`, `model_fits:0`.

## Required freeze before actual source execution

Root reviews the code, unit tests, this plan and prospective profile first.
Then writes a NEW method contract with schema
`h13-acceptance-support-method-v2`, `status:frozen`, exact profile file
path/SHA256 and `profile_identity=astra.io.digest(profile)`, all running
package sources plus astra.io, tests/plan and immutable source evidence, and
the complete requested input closure. Its `identity` equals
`astra.io.digest(body_without_identity)` and its `.yaml.identity` sidecar.
Source/input hashes are verified before decoding or reading origin keys;
file-hash caching uses astra.io's stat-keyed LRU. Profile `source_proof` binds
the exact plan and test-file identities; the reader requires both even if a
method body/identity/anchor is re-canonicalized. Essential gates must remain
exact integers, not booleans or coercible strings. A fresh output path is
required. Merely changing prospective status does not create a frozen method.

After root freezes and authorizes the actual support run, the explicit command
is `python3.13 -m astra.representation_acceptance_support_v2 --profile PATH
--frozen-method PATH --output NEW_PATH`, from AstraResearch. There is no
automatic invocation or workflow integration.

Small synthetic tests cover closure availability, equality callback order,
EOF right censor, future append, unknown/mixed and formation-cluster prints,
visit leave-and-return ownership with identical marginal prints/final pair,
completed memory at30s, freshness/status/day boundaries, quantity scale,
ULP identity/off-grid rejection, full-ladder crossing, raw ABI/clamp, repeated
landmark counts and independent frozen-plan integrity. Synthetic fixtures
validate code semantics only; none are research market data or backfills.

## Later predictive falsification, outside this support study

The direction hypothesis is that, at comparable path/age/marginal work,
departures from a bilaterally tested landmark differ in reversal toward that
landmark from unjoined marginal references. Competing explanations are
turnover, dwell/survival selection, endogenous common news and ordinary price
memory. There is no separate future magnitude claim. A complete retained
representation must still meet the unchanged V3 joint/magnitude/direction
rule. No claim-specific evaluator/head is adopted here.

Only after support and actual native prefix/parity pass can root freeze a
new FE comparison. Proposed bounded arms are B, B+H13, B+unjoined marginal
control. The control retains each side's original queue reference/work,
source timing/admission/censoring and origins, removing only shared visit
ownership. Synthetic counterexamples and existing neighboring families are
required novelty controls. Formal admission uses the existing frozen judge;
any control inference not authorized by that procedure remains descriptive
until an independent methodology is validated and frozen. H10 results must
also be interpreted before deciding whether adding a second module is
necessary. No FE comparison or acceptance is completed by this plan.
