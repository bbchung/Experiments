# Native H10 pilot and baseline invariance

This is a native semantic/coverage pilot, not an FE predictive comparison. V1/V2
science remains frozen. No model fit, label regeneration or full-universe replay
is authorized by preparing these files. Root runs native work serially after the
new module and focused tests have been built.

## Fixed TRAIN scope and source alignment

The already frozen TRAIN-only eligibility rule selected 2308 and 2317. Replay
exactly those symbols on 20260119 and 20260120. Missing files or metadata skip the
fixed cell without any replacement. The four source native partitions exist and
each has 1,381 original periodic origins.

Prepared configs and receipts are ignored under
`runs/information_state_20260930/native-h10-ordered-reset/`. The original
ordered-only preparation remains under `native-h10/`;
`native-h10/prepare_ablation.py` writes the updated four
`pilot/<day>/<symbol>/config.yaml` files and their narrow
`expected-origin-keys.parquet`; it does not launch coco. `preparation-receipt.yaml`
binds config/source/expected-key hashes and records exact command argument arrays.
Only the three native key columns were read from the parent partitions, with no
label column reads or model work.

The target retains its original `TwseFilter.0` with `RequireTradable: false`,
raw subscribed `CurrentBook.0`, `DemandRepairRenewal.0` ordered,
`DemandRepairRenewal.1` reset and two writers. Both instances
depend on the same filtered target book/trade feed. Their status filter is omitted
so native TRIAL/AUCTION/SUSPEND boundaries remain visible and censor state. Fixed
parameters are `MaxBookAge: 5s` and `MaxDemandGap: 30s`; these have distinct freshness
and episode-gap responsibilities. The only difference is
`ResetAtTouchRenewal: false/true`. The control drops the retained repair mark at
the first post-repair touch change; a later legal contained execution can seed a
fresh episode. This tests causal continuity across price renewal. The other
fifteen symbols retain only their
original `CurrentBook.0` subscriptions, preserving the parent's complete market
clock and loader merge order. No other large feature family is instantiated.

The original global ContractImporter/TradeBookMd configuration and raw paths are
preserved: `/mnt/data0/contract/tse/stock` and
`/mnt/data0/marketdata/tse/kgi/stock`. Replay reads each full native trading day,
including the original pre-09:10 warm-in; no raw timestamp cutting, invented rows
or replacement days. Daily engine construction resets H10 state. There is no
cross-day feature history in this mechanism.

The primary writer uses the original `PeriodicSampler` exactly: 10s, 091000
through 130000. The native repeat includes its endpoint, giving 1,381 rows. Require
exact agreement on all five keys `(day, symbol, SampleTime, SampleBookTime,
SampleBookSeq)`, including integer sequence and timestamp bits. A primary writer
subscribes the raw target book as the original did; sampling is independent of
H10 availability, phase, labels and prediction. **No Labelers** are configured;
later FE comparison must reuse original float64 label material and its existing
eligibility/split/judge.

Both writers export only eight H10 Alpha fields per instance: direction, log_work_strength,
signed_after_containment_share, signed_after_repair_share, delayed_repair_fraction,
signed_renewed_work, mid_progress_5ticks and categorical phase. Named metadata
retains raw CurrentBook OriginMidPrice/MidTicks/BidTicks/AskTicks and seven
non-Alpha H10 diagnostics: processed cluster exchange/availability time, anchor
and renewed raw price, cumulative complete/buy/sell chain counts. Diagnostics are
named `H10_ordered_<field>` and `H10_reset_<field>`. Raw price/time
diagnostics are audit metadata, never pooled model covariates.

The secondary event writer uses `BookFlipSampler` all_book, MinInterval 0s and
the same session bounds, writing to `events/` instead of `data/`. Its event rows
are diagnostics only and never replace or join the frozen evaluation origins.
They permit phase/counter/availability inspection at physical book cadence rather
than manufacturing a challenge from the 10s snapshot. Trade-only closures remain
observable only after their actual callbacks, recorded in diagnostics; the next
book row can inspect that already completed state.

## Causal acceptance before feature materialization

The native module buffers the complete current exchange-time cluster until the
next distinct exchange-time callback. State is published at that next callback's
received time. Whole mixed/unknown clusters censor; final open clusters are not
backdated or completed by end-of-day release. Hard noncontinuous/invalid boundaries
censor immediately. The online native path may consequently differ from the
conservative Python proxy; do not assert identical 52-chain counts without matching
observation scope and reset rules. Native RESUMED state retains its repair mark,
whereas the counting proxy starts a new candidate after completion. The proxy
also processes the full raw-day prefix; its active session condition gates
counters, not episode initialization. Scoped counter subtraction can include
chains that began before 09:10 and is not a cold-start experiment.

The owned global LogicalClock changes only the published view to `stale`/NaN
when the last closed known demand is older than 30s; it never closes a pending
exchange cluster or backdates newly received input. Exact 30s remains finite and
30s+1µs is stale. A later valid closed demand can establish a fresh current view,
while late-cluster availability is checked before any finite publication.

Check native completion status, 1,381 original keys per cell, exact float64 raw
OriginMidPrice agreement, no diagnostic availability later than its origin, and
the focused ordering/ambiguity/gap/visibility tests. Known inactive numeric values
must be zero, warmup -inf, censored values NaN. Verify positive integer event
support and genuine delayed quote-only repair before distinct new-touch execution;
anchor returns and repair after earlier renewal cannot manufacture completion.

Inspect event support and sampled support separately, including phase frequencies,
known zero versus missing, both directions and active/reset
lifetimes. The four-cell proxy support threshold is not a native parity assertion
or a substitute for these semantic checks. Before a later V3 comparison, freeze
the native binary/source/config/module metadata, reusable H10 material provenance
and the unchanged shared evaluation contract. Starting a new fresh baseline is
required; do not reuse V2 fitted-model outcomes as a new comparison baseline.

## Existing-feature controls

Root preserved the pre-H10 binary at
`runs/information_state_20260930/native-baseline/coco`, SHA256
`69b5123804b321e9f09f2c3b42511a232f2dacbc519688754282fc63c73f6c34`.
Preparatory hashing verified it against its existing receipt.

For 20260203 symbols 2609 and 2337, copy the immutable native_export_day's
**original config bytes** into separate `parity/{preserved,new}/<symbol>/` folders.
These original configs include their original features/labelers and clock carriers;
H10 is not instantiated. Every output path resolves in a fresh work folder.
Config hashes and original published values.parquet hashes are recorded before
the new build. Exact preserved/new command arrays are in the preparation receipt.

First compare preserved-binary replay bytes with the original published parquet;
then compare the new binary with both preserved and published output. Matching
preserved/new alone cannot establish source correspondence to historical parent
material. If the first comparison fails, report that pre-existing source/binary
correspondence gap separately before attributing any difference to H10. No original
artifact/config/output is mutated. Do not compact, recast or rewrite these parquet
files before the required byte comparison.

## Bounded full-material cost, contingent on pilot success

The frozen role calendar is **161** unique days: TRAIN102, ES9, calibration20,
development30. Do not silently expand this to 163 or include observed later dates.
Stat-only inspection found all 2,576 role-day×symbol raw files available, totaling
**3,995,918,965 compressed bytes (3.72GiB)**. Missing future input cells continue to
skip without backfill. These are file metadata counts, not another raw scan.

For eventual feature-only H10 material, use one sequential native replay per day
with all sixteen ordered/reset H10 pairs/writers; it reads the market universe once
and shares those origins with the parent baseline/labels. Target-sharding while
retaining all sixteen clock carriers would repeat about **59.54GiB** of compressed
I/O. Export the one new representation once and reuse it for complements and
causal ablations rather than rebuilding the baseline or multiplying the
feature×horizon×symbol matrix. The coordinated full producer root is
`native-h10-ordered-reset/full/<day>/data/<day>/<symbol>/values.parquet`.
Native output size and wall time remain unmeasured;
estimate them from the four-cell native pilot before full materialization.

## Completed pilot and full-day preparation boundary

Independent validation of all four completed native cells passes original int64
keys, float64 mid bits, phase/sentinel semantics, origin availability and the
predeclared 30s support thresholds. The immutable old screen is **still failed**
because current/original historical byte correspondence fails. Preserved/current
versus new H10 replay invariance passes separately. Historical after-freeze state
corrections are documented by the original SOURCE_AUDIT.md; no original feature
value is translated or overwritten. Component provenance requires its own
validated frozen method before an FE comparison can proceed.

The ignored `native-h10-ordered-reset/prepare_full.py` only prepares fixed role-day
configs, argument arrays and content hashes. It emits one daily config containing
all sixteen original groups and ordered/reset pairs, one periodic writer per
symbol, no event writer and no labeler. BasicInfo, raw source preference,
calendar, template config, binary, runtime and build-source receipt are bound.
Higher-priority native raw-file alternatives must remain absent; changed loader
source preference cannot bypass the frozen input manifest. Missing cells or days
are recorded and skipped without replacements.

The first 20260119 monolithic replay must serve as both the benchmark and its
reusable full artifact. Validate all sixteen original keys and physical mid bits
before running the remaining 160 days: four one-writer pilots cannot alone prove
that sixteen writer timer registrations preserve every parent origin. Preparing
these files launches no native engine or fit and does not itself authorize full
materialization.

Actual preparation found the native loader selects `.bin.zst` for all 2,576
cells, before its CSV fallback. Those exact native sources total
**3,274,088,189 compressed bytes (3.05GiB)**; their hashes and the 161 original
BasicInfo files are in `full-preparation-receipt.yaml`. The earlier 3.72GiB
estimate remains explicitly a CSV stat estimate. Python CSV support and actual
BIN native replay are separate evidence, with no assumed event parity.

Root's first monolithic 20260119 replay passed original keys and float64 mid bits
for all sixteen symbols. Its completed proof is
`full-first-day-validation.yaml`; the serial `execute_full.py --remaining` route
requires the same plan/script/lineage/support bindings and never overwrites
existing output. Every child runs to successful native completion before another
day starts.

## Actual receive-prefix evidence

For fixed 20260119/2308, the earliest supported real 30s origin has
R=1768784999667336 < S=1768785000000000 <= D=1768785000037455 <
A=1768785000308303. R is the last target receive in an open exchange cluster;
A is the next distinct target exchange callback's actual availability. Carrier
2409 supplies the actual received event at D. These are genuine native source
timestamps, without generated clocks or rows.

The cut retains all sixteen BIN streams through each loader's clamped receive
<=D, including their full original warm-in. Every original native header/payload
byte, source order and sequence is retained; compression changes only the
container. Original BasicInfo remains the importer source. Root replayed that
fresh prefix and its one common periodic origin retained **all 37 output fields
bit for bit**, including H10 numeric/phase/diagnostic fields and raw mid. No
prefix origin was emitted after D and the open target exchange cluster was not
backdated into S. This is a bounded temporal causality proof, not predictive
evidence or a claim about unseen dates.

The first live-source receipt `receive-prefix/validation.yaml` is preserved.
The final `validation-snapshot.yaml` independently repeats the same checks and
binds all 1,637 immutable compiled-source copies to the original prebuild hashes,
including the four exact native ABI/parser source contracts. Both recorded
validations pass; future C++ edits cannot alter the already compiled producer's
source evidence. The new validator and its explicit dependency closure will be
frozen into V3 before any FE comparison.
