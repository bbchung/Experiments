# H16 raw native source support, prospective version 3

This is a prospective native implementation and support procedure for
`AuctionCycleRawInformation`, under `H16_RAW_PLAN.md`. It is label-free source
support research, with no FE admission, prediction comparison or final OOS
claim. All H16 V1/V2 code, profiles, plans, frozen methods and outputs remain
immutable. V2's 32 exact key matches, four registration byte matches and two
failed baseline mid cells remain recorded failures.

Root alone calls the lifecycle APIs in
`runs/information_state_20260930/native-h16-auction-cycle-v3/native_pilot.py`.
The draft helper does not run replay commands. Its APIs are
`prepare_metadata(binary)`, `producer_payload(preparation_path)`,
`preflight(producer_method, job=None)`, `bind_evaluation(producer_method)`,
`verify_evaluation(evaluation_method)`, `check(evaluation_method)` and
`freeze_payload(payload, destination)`.

The native routing preserves TwseFilter with `RequireTradable: false` and
`StatusFilter: 'TRIAL || !TRIAL'`, CurrentBook, original raw MD carrier order,
all sixteen symbols, raw mid metadata and the original ten-second periodic
sampler from 09:10 through 13:00. CurrentBook's existing declaration is copied
unchanged. A private CRIT producer tap sees raw Book and Trade before HIGH
TwseFilter; it does not change messages, native feed dispatch or CurrentBook.
Seven Alpha fields, comprising five numeric and two categorical, are native
`AuctionCycleRawInformation.0.<field>.0`. Twenty-seven native Info fields are
exported as `H16_RAW_<field>` metadata and never enter predictors. The complete
schema has 41 columns: three exact int64 sample keys, seven Alpha, four original
raw CurrentBook fields and 27 float64 Info fields.

Raw source receive time R and actual publication availability A are separate.
Public source operands require R <= A <= original SampleTime S. Reported own
exchange E is an independent clock; there is no E <= R, A or S requirement.
All clock and integer diagnostic identities are nonnegative exact integers
strictly below 2**53, with exact positive-zero absence identities. Freshness
uses S minus source R, not S minus delayed consumption A. Current residual basis
uses matched Trial and Book own E and fresh source R on both operands. Zero-mass
Trial with unavailable price retains exact known +0 mass and unknown price
fields. Negative zero, positive infinity, unavailable-to-warmup collapse and
unsafe floating integer identities fail validation.

A bounded raw queue of 256 records has explicit native overflow accounting.
Every original ten-second origin in every cell must show zero cumulative raw
overflow. Overflow never qualifies as zero activity or as ordinary unavailability
that may be silently filtered. No availability-based sampling or row removal
is allowed. No EOF callback, timer or expected auction deadline invents a
source event. Actual match intervals remain strictly past source-first-R gaps;
match mass is the observed as-of same-E sum, not a claim of a complete batch.

The fixed native pilot uses 2026-01-19 and 2026-01-20 and all sixteen original
carriers. The six required batch cells are symbols 3481, 2344 and 2337 on both
days. Each requires at least five distinct session actual cycles, five distinct
sampled source cycle identities and twenty fresh strict-post-anchor support
origins on the unchanged original thirty-second subset. Repeated prints do not
create cycle credit. Other symbols retain their original rows and typed regimes;
they have no mandatory batch-cycle support quota. All 32 cells require exact
original three-key and float64 raw-mid bit parity, 1381 ten-second rows and 461
original thirty-second support origins. A missing fixed BIN, BasicInfo, contract
or original parent artifact is skipped without substitute; the complete gate
then cannot pass. RequiredAbsent checks run before expensive closure checks.

Per-field known-zero/nonzero/positive/negative, finite, warmup, unknown and
finite-bit diversity are descriptive under three predeclared masks: all original
thirty-second origins, observed auction/indicative origins and session cycle
support origins. Price, quantity, matched-E, interval and historical-error
operand support are reported separately. A constant historical acceptance error
or an individually degenerate feature does not reject the complete mechanism;
any later field redefinition or removal requires a new prospective contract.

Root first builds/tests the new C++ leaf while preserving all old native sources.
Preparation captures the new Release binary, four new raw material source files
and only explicitly permitted build metadata changes. Unchanged immutable H16 V1
compiled snapshots are reused instead of copied. The live old source hashes are
verified before and after capture and again by preflight. The old 51-library
runtime is shared; immutable recorded system loader aliases are resolved to
actual targets for byte checks without rewriting those records. The prospective
profile belongs to both source and input closures. Own helper, tests, root
executor, plans, io kernel, original source snapshots, native binary, runtime,
original daily metadata, raw input identities, BasicInfo and configurations are
closed before replay. No large aggregate dataset manifest is parsed.

Root freezes `h16-raw-native-producer-method-v3` before any native command. One
Jan19 registration-only job reuses the exact old H15 pair configuration and must
match all four old data/events Parquet files byte for byte. Only then may the two
fresh all-sixteen raw pilot jobs execute. Each job has a fresh work directory,
immutable command and producer-bound receipt. A completed job cannot be resumed
or overwritten. Metadata evaluation binding hashes completed native artifacts
and receipts without reading any native table values. Root freezes
`h16-raw-native-source-support-method-v3` before `check()` decodes the narrow
native and original key/mid projections. Re-canonicalizing a manifest does not
permit omission of profile, source, raw, runtime, parent or execution closures.

A source support pass is separate from an actual real receive-prefix proof.
The first prefix draft has never been selected, frozen or replayed; its next
version must use these 41 fields and preserve original filters. Root must freeze
its source-only selector and comparison before the corresponding actions. Native
source support and prefix causality still do not establish FE predictive power.
