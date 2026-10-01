# Independent native clock-domain methodology study

This study resolves an integrity-validator assumption before any V3 FE freeze
or CatBoost fit. It uses source semantics, a real raw BOOK witness, the complete
already-produced native diagnostic population, controlled validator tests, and
an independent native exchange-offset metamorphic test. It reads no labels or
model scores, launches no fits or replays, and changes no representation values.
The existing frozen component-lineage V2 validator and its historical receipts
remain unchanged. Its pilot evidence remains valid for those four native cells;
its `exchange <= available` assumption is not a general clock-domain contract.

## Hypothesis and source proof

The vendor exchange timestamp and receiver timestamp are different clock
domains. The hypothesis is that publication availability belongs to the receive
domain, while exchange ordering and book/trade matching belong to the vendor
domain. Consequently `available <= origin` is required exactly, but neither
`exchange <= available` nor `exchange <= origin` follows from causality.

`src/marketdata/parsers/twse_parser.cpp:245–265` derives exchange time from the
wire and retains the independently supplied receive time, substituting exchange
only when receive is absent. `src/oms/model/quote.cpp:519` advances the logical
clock using receive time. DemandRepairRenewal's `advance`, `close_cluster`,
`process_book`, `on_trade`, and `refresh_origin` use independent within-axis
ordering/age checks: the previous exchange cluster becomes available only at the
receive time of the next distinct exchange callback. Five-second freshness is
checked in each domain, and the thirty-second episode age uses receive time.
There is no source requirement that exchange time precede receive or origin.

`docs/feature_family_abstraction_v2_review.json:11474–11478` explicitly treats
receive-minus-exchange as feed timestamp offset, without inferring synchronized
clocks or transport-only latency. `docs/pre_model_feature_research_20260926.md:68`
states that receive order determines availability and exchange regressions are
separate. Existing PressureResponseState, TradeClustering, SyntheticMeasurement,
and signed CrossMarketBasis timestamp context corroborate the same distinction.
Root binds the exact compiled source copies and the independently inspected
documentation into the frozen source closure before using this method.

## Fixed validation rule

Native origin keys retain nonnull `int64`; H10 diagnostics retain nonnull
`float64` exact integer microseconds. All positive timestamps are strictly below
`2**53`. No fractional clock, infinity, NaN, negative timestamp, negative zero,
rounding, clipping, tolerance, or timestamp translation is accepted. Exchange
and available zero states agree; a state explicitly requiring an observed closed
cluster requires both positive clocks. The unobserved state is paired positive
zero. Receive availability must be at or before the origin using exact integer
comparison. Vendor exchange time has no comparison against the receive-domain
origin or availability. Sampled global exchange monotonicity is not imposed:
censoring/rearming can reset source state, and snapshots are not an event stream.

No FE sampling, labels, splits, training procedure, metrics, or acceptance rule
changes. All V3 arms must use the same frozen temporal integrity rule and the
same unchanged original rows/labels/feature values after this independent method
is validated and frozen. The old cross-domain failure is recorded, not erased.

## Predeclared validation and rejection

The complete fixed population is 2,576 native cells, 3,540,544 rows per variant.
The independent census reads only the three sample keys and four native clock
diagnostics. Every cell must have exact nonnegative clock storage and paired
positive zero; every variant must have zero future receive availability. Census
totals must reconcile with every declared cell, with no omission, duplicate,
date before 2026-01-01, or altered producer-journal/projection identity.
Exchange-ahead counts are descriptive; they cannot relax receive availability.

The observed 2026-02-11/2481 row 76 has exchange
`1770772959962714`, available `1770772959957872`, and origin
`1770772960000000`. The raw unchanged source contains BOOK ordinal 7699 at
that exchange time and ordinal 7700 at the next exchange time, both received at
the availability timestamp. Their actual source hash and decoder source are
bound. Thus the prior exchange cluster closed 42,128 microseconds before the
origin despite vendor exchange being 4,842 microseconds ahead of receive.

An independent native unit test shifts all positive exchange timestamps by
`+3,600,000,000` and `-3,600,000,000` microseconds while preserving receive,
status, callback order, prices, and quantities. All seven numeric AlphaFactor
bits and the phase must remain identical for both ordered and reset variants;
only the exchange diagnostic shifts. It covers active stage progression,
receive-origin staleness, pending-cluster closure, unknown-trade censor/rearm,
and hard status. Root builds/runs it once, records command/log/test binary and an
immutable test-source copy, and binds the result. No feature/timestamp transform
is applied to real data. A failing native invariant rejects the new method.

Python failure-mode tests must reject even a one-microsecond receive violation,
unsafe clock identity, zero-state disagreement, scope/evidence omissions, source
drift, and frozen profile/anchor mutation. They must accept positive and negative
vendor clock offsets without altering input arrays. Toy validation conformance
alone is insufficient: raw witness, complete census, and actual native unit
receipt must all pass before freeze.

Root freezes `native-temporal-method-frozen-v1` with a canonical self-identity and
separate identity anchor, including the profile, plan, tests, running temporal
kernel/hash implementation, source audit, raw witness/source, census/source and
producer-journal bindings, and native metamorphic receipt/source/binary/log.
The profile declares every required source-proof path explicitly, including the
inspected immutable producer source copies and documentation. Native unit
evidence must confirm the actual producer binary is unchanged. Verification
reconstructs all mandatory evidence and demands an identical validation payload
and every required file in the same source/input scope, even if someone rewrites
a canonical manifest and its identity anchor. Witness origins must retain their
native integer type before conversion; fractional timestamps cannot be truncated.
Only afterwards may a NEW V3 integrity adapter import the frozen rule and bind
the method manifest/anchor; old frozen provenance/FE sources remain untouched.
This proves integrity-method conformance, not predictive FE improvement or OOS.
