# H16 native pilot protocol v1

This is a prospective draft for the implementation contract in H16_PLAN.md.
It provides label-free native support and causal integrity evidence, without
FE admission, a trained model, a changed universe, or pristine OOS evidence.
Root freezes the producer method before commands and freezes the evaluation
method before reading native feature/count values. Existing H15 mapping
failure and all earlier producer/support/prefix contracts remain immutable.

The dates are TRAIN 20260119 and 20260120. The original16 carrier order is
2330,2317,2454,2308,2382,3231,2603,2609,2615,3481,2409,2344,2337,2481,
3037,3711. Every original source is retained, including continuous stocks.
There is no eligibility selection or row availability filter. Missing fixed
BIN/BasicInfo/contract skips the day, while missing original key/mid artifact
skips its writer. No replacement date/symbol or synthesized input is allowed.
RequiredAbsent is checked before hashing any larger closure. A missing cell
is reported and prevents complete32-cell support acceptance.

Each present contract directly subscribes AuctionCycleInformation to raw Book
and Trade with Symbol equal to GID, IndicativeMaxAge10s and the explicit native
accept-all expression `TRIAL || !TRIAL`. No TwseFilter is attached anywhere in
the H16 daily configs: its global Feed filter would otherwise discard valid
zero-quantity TRIAL disclosures even for direct subscribers. CurrentBook.0
and the original all16 raw MD/calendar/BasicInfo routing are retained.

DatasetWriter.0 samples every10s from09:10 through13:00 inclusive, preserving
all1381 original SampleTime/SampleBookTime/SampleBookSeq keys and native raw
OriginMidPrice bits. It exports the fixed5 numeric and2 categorical Alpha
fields plus22 Info metadata aliases and4 CurrentBook raw metadata fields.
There are no labelers, events-only replacement origins, Python features,
quantity cutoffs, additional samplers, or feature parameters.

Root builds and tests the new C++ producer, then prepare_metadata captures
the fresh binary and compiled source capsule. Existing native math must be
unchanged from the613260 H15 producer; only new H16 material sources and the
explicit build bookkeeping/appended test target may differ. The51-dependency
runtime is shared and its copied and current-system identities are checked.
The new module declaration must match the native feature-guide exactly.
The capsule records compile-command presence and before/after live hashes;
this does not substitute for root's actual build/test evidence.

The first native command is registration_control on Jan19 using a byte-exact
copy of the existing H15 pair config. All four old pair data/events Parquets
must be byte-identical to the frozen613260 outputs before either fresh H16
daily command may run. The control is not an H16 support scan. Its existing
raw source order, calendar, runtime, source config and expected outputs are
bound. Every native command runs to a fresh work directory and writes a
method-owned execution receipt binding job, command, exit0, log and ordered
output records. Completed commands cannot be relaunched or overwritten.

After commands, bind_evaluation hashes the exact outputs/receipts/status
files without decoding native values. Root freezes that evaluation payload
with its profile, producer method/anchor and mandatory source/input closure.
Only check(frozen_evaluation) then decodes the36 declared columns plus original
3 keys and OriginMidPrice. Original daily native manifests are small direct
lookups; no aggregate298MB Store.resolve is used.

All present cells must pass exact typed schema, role, no-NULL, positive-zero,
integer/clock range<2**53, receive availability<=origin, past-interval identity,
partial operand and category checks. Raw exchange clocks are an independent
axis: E<=receive/origin is not required. Hard boundaries are actual received
facts; clocks do not manufacture anchor/group closure. Each column preserves
warmup−inf, post-observation NaN and known-zero/finite state. Continuous
observations are typed inactive and exact+0. Unavailable price/trial quantity
operands stay NaN. Trade-before-matching-Book can publish F1/F4 while F2 is
NaN; finite F2 requires the same ownE disclosure. Historical error/age may
remain finite independently of current trial availability.

Formal support counts ONLY the original46130s origins per cell. Each of
3481,2344,2337 on each fixed date requires at least5 new observed actual cycles
inside[09:10,13:00), at least5 distinct actual(E,firstR) cycles represented by
native cycle_support at those origins, and at least20 fresh strict-post-anchor
support origins. SameE fragments/repeats cannot add distinct-cycle credit.
Session cycles use the original13:00-minus09:10 native counter convention;
the lower snapshot's already-visible anchors are excluded conservatively.
Other stocks have no cycle minimum and remain in the output/cohort, with
regime/category counts reported. Quantity, price, interval and each numeric
field's variability are reported separately. A constant acceptance-error
field cannot automatically reject the entire mechanism; any later deletion
or redefinition requires a new version/contract before predictive comparison.
Numeric summaries include exact positive-zero, finite nonzero, positive,
negative, warmup and unknown counts under three fixed descriptive masks:
all original30s origins, observed auction/indicative regime origins, and the
session cycle_support origins actually admitted for this gate. Native operand
presence, matched Book/Trial E, historical error and observed interval counts
use the same masks. Continuous inactive +0 does not earn mechanism support.
These masks only describe unchanged native fields; they never filter the
output cohort or form Python features.

Final support requires all32 original cells and all six fixed batch-cell
gates. A supported pilot still requires a separate real receive-prefix proof
and a newly frozen FE evaluation contract before predictive research. No
label/score/model is read, no model is fitted and no predictive success is
claimed in this protocol.

Root API sequence:

1. `prepare_metadata(new_binary_path)` writes source/preparation/config
   metadata and returns producer_payload. No native commands are launched.
2. `freeze_payload(payload, frozen-method.yaml)` is root-only after code,
   profile and tests are READY/suspended.
3. `preflight(method, job)` before each exact command, starting with control;
   root writes each execution-receipt.yaml after its actual command.
4. `bind_evaluation(method)` returns a draft evaluation payload without values;
   root freezes it with freeze_payload before any check.
5. `check(frozen_support_method)` validates and writes a fresh
   source-support-validation.yaml. It never launches a replay or learner.
