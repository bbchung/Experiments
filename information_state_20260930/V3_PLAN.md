# V3 native H10 ordered demand versus touch-renewal reset

Status: independently prepared, unfrozen comparison recipe. Root alone may
materialize, build, replay and train after the separately validated component
gate's applicable checks pass. No current predictive admission is implied.

The new hypothesis is strict known execution followed by later quote-only
repair, retention of the repaired mark through touch renewal, and later
same-side known execution at the renewed touch. Existing pressure/refill
aggregates already express many related mechanisms; the exact new information
is the ordered linkage. H10_NOTES.md records the prior support screen and
existing producer boundaries. The reset control uses the same producer,
feeds, gap/age policies and normalization, with ResetAtTouchRenewal=true.

The three primary 300s arms are a freshly fitted full legacy baseline,
baseline plus ordered H10, and baseline plus reset H10. There are exactly
three CatBoost GPU0 fits, scheduled serially under the global lock; no HPO or
secondary horizon fits enter this bounded mechanism comparison. The 60/120/180s
native labels still bind the common complete-horizon cohort. No FE result
changes sampling, labels, cohort, train/es/calibration/development roles,
training, metrics or acceptance rules. The V2 primary300 judge remains exact.
The independent claim-specific methodology prototype is not adopted here.

Each H10 arm adds these seven numeric fields:

```
direction
log_work_strength
signed_after_containment_share
signed_after_repair_share
delayed_repair_fraction
signed_renewed_work
mid_progress_5ticks
```

The eighth field is the native string category `phase`. Timestamp, price,
episode counters and other diagnostics are excluded from predictors. Both arms
use canonical H10 column names; the frozen arm recipe/cache identity selects
DemandRepairRenewal.0 or .1. Numeric -inf warmup, NaN stale/censored/uncomputable,
and finite active values remain unchanged through the same float32 consumer
casting used for baseline. H10 inactive values are exact positive-zero bits;
active values are finite with direction exactly +/-1 and positive work strength.
Baseline's generic permitted +inf contract remains unchanged. The native phase
strings warmup, inactive,
contained, repaired, renewed, resumed, stale and censored are categorical,
never converted to ordinal numeric codes. No feature-availability row filter,
zero substitution, epsilon repair or forward filling is permitted.

Feature files are joined on day, symbol, SampleTime, SampleBookTime and
SampleBookSeq. Every original observation in the declared role dates must
exist with its exact native book identity; extra native rows are harmless.
Missing original data cells are absent from the parent's anchors and receive
no replacement. A missing required newly exported cell is an integrity failure,
not authorization to alter the parent's frozen comparison population. The
candidate loader reuses the inherited baseline population and common mature
labels, including the original held-symbol restrictions.

The fresh baseline and both candidates receive identical original full
numeric/nominal context, weights, recency half-life, early-stopping role,
MultiClass loss, parameters and seed. New local fit/score functions are exact
source copies of the inherited functions; targeted structural parity and
orchestration tests guard the necessary categorical extension. No frozen V1/V2,
existing CLI or original signal profile source is edited.

That full control comes from the immutable historical producer (SHA256 starts
2374), not the current engine. The current baseline producer (69b512) and H10
producer (8678aa) are byte invariant in the separately replayed baseline
control, while the historical-to-current all-byte comparison fails only in
the nine already corrected FlowResponseSurprise warmup states and fourteen
already corrected CrossReturnContext halt fields. SOURCE_AUDIT.md in the
fe_origin_20260930 experiment records those intentional corrections after its
predictive freeze. V3 retains the original historical values of these 23
fields in every arm. The H10 overlay uses its current producer; this component
boundary does not create baseline FE gain or repair the historical baseline.
Historical all-byte parity failure remains explicit provenance evidence.

Before any V3 preparation or freeze, a separate no-FE producer reproduction
method must be validated and frozen. Its native-component-lineage-validation-v2 receipt
must confirm historical parent reproduction and current-control invariance,
bind its method identity and all three producer binary hashes, and hash the
checker sources, binary/config/raw inputs, frozen method manifest/identity
and original parity reports. V3 closes this receipt and its dependencies;
it does not substitute whole-producer equality for component lineage.
`method_contract_path` locates the manifest without hardcoding a study-plan
filename. Its `identity` must equal astra.io.digest of the body excluding
identity, the receipt's method_contract_identity, and the immutable `.identity`
anchor. The receipt must bind both files and the complete method source/input
closure. A post-freeze method mutation cannot become valid by changing only
receipt hashes.

Calibration decisions nominate at most one candidate before any development
score is computed. All development arms are reported. Final support retains
the exact existing paired-day, nonoverlap, event-support and seen/held-group
guards, plus prior nomination. Ordered versus reset is a predeclared same-judge
ablation report on both roles; it creates no additional nomination. A lack of
clear ordered-versus-reset evidence cannot establish ordered chronology as
the source of any baseline improvement.

H10_NATIVE_SCREEN.md was fixed before the first replay with SHA256
6b9899a14d87541cd71d171d5355d7c5929bb0db5a8a5ea202f686eccf512db1.
Its original whole-producer byte-parity gate failed on the already documented
historical/current corrections; that failure is retained and is not relabeled
as a pass. Native chain/sampling/sign evidence does not erase this failure.
The independently researched, validated and frozen component method and new
H10_COMPONENT_GATE_V2.md permit qualified materialization under explicit
component lineage. This new gate was declared before full replay and requires
all 16 first-day symbols' exact keys/mids, full-cell phase/clock validation,
and a real receive-prefix straddle test before V3 freeze. Both gate versions
belong to the source closure. FE sampling, labels and predictive judge remain
the inherited contract.

The native preparation receipt is pre-build evidence only. Root must create
`native-h10-ordered-reset/validation.yaml` after the relevant native checks:

```
passed: true
binary_sha256: SHA256_OF_IMMUTABLE_NATIVE_BINARY
sources:
  - {path: ABSOLUTE_CPP_OR_HEADER_PATH, sha256: SHA256}
inputs:
  - {path: ABSOLUTE_RAW_CONFIG_CALENDAR_OR_PROVENANCE_PATH, sha256: SHA256}
```

Source and input records must match the replay that produced the validated
parquet. The binary is an immutable snapshot at
`native-h10-ordered-reset/coco`, not a later mutable build output. Preparation
checks those bindings and reads each cell once for both variants. It shares
existing context caches through hardlinks; no full baseline dataset is copied.
Every original full-scope observation also compares OriginMidPrice as exact
float64 bits to the inherited mid cache at its original parent row position.
Both variants must preserve declared phase/numeric sentinel states and exact
nonnegative integer-microsecond closed-cluster clocks strictly below `2**53`,
with paired positive-zero unobserved states and positive clocks for active
observed states. Receive availability must be at or before origin exactly;
exchange time is in the independent vendor domain and has no required ordering
against receive availability or origin. Negative-zero clocks, unsafe/fractional
timestamps and changed numeric types are rejected. The five narrow validation metadata columns are not
predictors. No endpoint labels are read during this validation. The preparation
receipt must attest all original cells passed before freeze is permitted.
The narrow Arrow read rejects storage nulls and changed numeric/key types before
pandas conversion; IEEE NaN remains legal in H10's explicit missing phases.

The first full preparation failed the former `exchange <= available` guard
at 2026-02-11/2481. This historical failure is retained. Before changing any
V3 guard or running an FE fit, a separate temporal-method study inspected the
parser/producer clock semantics, bound the actual raw received BOOK closing
pair, censused all 2,576 already-produced cells, and ran an independent native
exchange-offset invariance test. Both variants have zero future receive
availability across 3,540,544 rows, while vendor exchange is ahead of available
in 30,802 rows and ahead of origin in 5,268 rows. All seven numeric bits and the
phase remain unchanged under positive/negative one-hour exchange-domain shifts;
the native producer binary and real data values remain unchanged. This is an
integrity-method study, with no label/model-score reads or FE outcome selection.

That method was validated and frozen with identity
`182d8d530e8f75fdb1861b6ee6a3a9b99971763fbcbdf5bd7855e9a243304255`
before this V3 adapter imports its exact receive-availability validator.
`TEMPORAL_METHOD_PLAN.md` and `temporal-profile.yaml` define its independent
source/witness/census/native-test contract. V3 requires the method manifest and
anchor, its exact predeclared identity, matching validation payload, current
H10 producer binary, and complete source/input closure. Omission, profile/source
mutation, binary drift, or altered validation flags rejects preparation/freeze.
The method manifest, anchor and validated receipt are also V3 freeze inputs.
The old frozen lineage validator and prior receipts are unchanged. No sampling,
labels, FE judge, representation values, or censor policies changed with this
integrity correction; all three arms use the same final frozen rule.

`receive-prefix/validation-snapshot.yaml` must use native-receive-prefix-validation-v1,
with passed, genuine_straddle_found, common_origin_present and
all_common_origin_bits_exact and compiled_source_snapshot_bound all true,
plus the immutable binary SHA and
explicit source/input SHA records. This receipt, its real original/cut BIN
inputs and native full/truncated outputs belong to the freeze closure. It is
an actual causal publication proof, not a manufactured toy timeline. The
earlier validation.yaml is retained; the new receipt binds immutable compiled
source copies and retains the original proof/ledger as input provenance.

V3's `features`, `preflight`, `freeze`, `verify`, `run` stages are exposed via:

```bash
python3.13 -m astra.representation_research_v3.runner \
  --profile experiments/information_state_20260930/profile-v3.yaml preflight
```

The no-fit preflight checks every arm's exact cohort against V2 and checks its
numeric/nominal schema with tiny CatBoost Pools. Freeze closes inherited V2
sources/inputs/profile/manifest/identity, new V3 sources/profile/tests/plan,
native C++/export/parser source dependencies, native configuration/raw/calendar
dependencies, binary, preparation/validation receipts, exact native parquet,
aligned caches, environment and preflight before the first fit. Modified or
resumed profiles/sources/inputs must be rejected. Root may update these new
unfrozen files during review; no later scientific change can retain the same
freeze identity.

Static scale audit provides a qualification, not predictive evidence. The
legacy full baseline contains 3254 numeric and 132 nominal fields across
173 numeric producer module types, all matching AlphaFactor declarations.
However, `PriceMemoryState::publish` directly outputs six log1p raw cumulative
volume fields (`price_memory_state.cpp:97`). Logging does not remove symbol
volume scale. The full legacy baseline remains an unadmitted scientific control.

Many existing scales are already semantic: signed flow/total flow (OFI), flow
over displayed depth (BookClockFlow/TradeCadenceState), quantity over previous
book depth (PriceFormationPath), slope over spread after depth normalization
(KyleLambda), relative same-symbol past volume (TradeVolumeRegime), depth
shares/HHI, and fixed cap-relative durations. Native tick distances/variation
have a common interpretation against the five-tick target; dividing all stock
volatility away would discard large-move base-rate information. Raw event-rate
levels and elapsed event-clock spans require explicit activity/horizon semantics
or a predeclared activity/tick-regime group. A text-screen footprint of 155
activity/rate/count-related columns across 20 producers is triage only: several
are already relative contrasts or normalized work rates. This is not evidence
that every such column is defective, nor a completed audit of all 173 formulas.

A compact future replacement should retain session/anchor displacement,
signed demand over displayed capacity, supply-repair response, regularity/live
demand evidence and previous-price memory in target-tick/causal prior-volume
units. Absolute target-scaled movement capacity and relative activity anomaly
serve different purposes. Such a representation must be preregistered and
compared from a fresh baseline; it is not mechanically applying z-scores to
3254 columns, and it is not the small 27-column core alone.

All historical development origins remain exposed. This comparison can provide
developmental representation evidence only. At least 30 newly available future
sealed dates, after candidate freeze, are still required for pristine OOS proof;
the current package does not implement or score sealed OOS.
