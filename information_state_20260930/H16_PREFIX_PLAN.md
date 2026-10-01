# H16 native real receive-prefix protocol v1

This is a prospective draft. Its author has performed no selection, metadata
binding, raw inspection, replay, feature-table read, label or model work. Root
alone executes it after the complete H16 native pilot support gate passes.
Pilot producer/support/profile/source contracts remain immutable. This study
proves a bounded native causality property, without FE admission or OOS.

The fixed day is20260119 and fixed target3481. All16 original carrier streams
remain. There is no alternate day, target, source or origin if selection fails.
No Alpha/category/quantity/price/label/score eligibility filter is used, and
no pending exchange group or unsealed mark is required. H16 actual-match and
proposal facts may be immediate observations rather than group closures.

Root calls bind_selection with the actual working H16 producer/support method
paths and optionally the support receipt path. Complete32-cell pilot integrity
and support PASS is mandatory. The method binds the new binary/source capsule,
all16 Jan19 full native outputs, original raw inputs, runtime51, pilot execution
receipts and the immutable native-h15-receive-prefix complete-spool receipt.
Every spool must correspond to the same original raw source and existing path.
Old/new compiled md_msg and TradeBookMd source hashes must agree with the ABI
reader. Existing complete spools are reused; no original compressed source is
decompressed again or copied wholesale. Root freezes this selection payload
before native keys or source headers are read.

Selection reads only original SampleTime/SampleBookTime/SampleBookSeq and
schema plus complete-spool source headers. Search ascending original30s
S>=09:10. R is the last target effective receive strictly before S; A is the
first target loader-dispatched record strictly after S; D is the earliest
global loader-dispatched record strictly after S. Native comparator ties use
effective receive, local sequence, Trade before Book, symbol and source path.
Choose the first R<S<D<A case; otherwise report unsupported and launch no job.
Empty Book clock/status publication and raw zero-quantity Trade are included.
The protocol does not claim every record emits a H16 Book/Trade callback.
Raw exchange clocks are independent; no E<=receive/origin condition is used.

D>S supplies a real input which can flush periodic S. Records at S remain;
S is the native pre-message snapshot. Cut ALL records with loader-effective
receive<=D, including every same-R record and entire warm-in. The loader's
per-stream max(rawR,previousR) clamp determines extent only. Header/payload
bytes, raw R/E, symbol, sequence and sentinels stay unchanged. No execution,
book, origin, group close or EOF/timer event is invented. Prefix config changes
only the TradeBookMd raw directory; all16 writers, parameters and sampling stay.

Header indexes contain only receive times/byte offsets within the old512MiB
spool bound and are shared in one root process. Cuts are fresh compressed exact
byte prefixes. Verification repeats the deterministic selector using these
indexes, checks count/extent/cutoff/source identity, and compares original byte
prefix digest with actual compressed cut content. Cut-content validation is
cached by immutable SHA; it does not decode new original sources or form FE.

Before replay or value comparison, bind each full native schema and original
KEY population<=D: counts, ordered key digest, schema digest and prefix path.
No native Alpha/Info/rawmid value is read by this bind. Root freezes comparison,
preflights, then runs its exact recorded native command once in a fresh workdir.
Execution receipt binds comparison identity, command, exit0, outputs, log and
completed status. No overwrite/resume or another job's result is accepted.

Configured output absence is allowed ONLY when its frozen original common KEY
count is exactly zero. No synthetic empty artifact is made. Target periodic S
is mandatory and cannot be absent. Ordinary nonempty periodic absence rejects.
Presence must agree with execution/filesystem identity. Full schemas and common
ordered key hashes are rechecked, rejecting dropped/reordered/invented keys and
any origin>D.

Only final check decodes H16 values: all common original exact keys in ALL16
cells and ALL36 native fields (3keys,5numeric+2categorical Alpha,4raw metadata,
22Info). Compare float64 uint64 bits, including NaN payloads,−inf,+0/−0; compare
strings exactly. Types, no-NULL, roles and schema metadata match. The unchanged
pilot validator enforces partial sentinels and clock/state contracts. Selected
S must occur at its exact original BookTime/Seq. No label, recast or Python FE.

PASS means that actual bounded prefix reproduces the common native information
exactly with every original stream preserved. It does not prove all histories,
predictive power, complete normalization, final OOS or trading performance.

Root-only API sequence:

1. bind_selection(producer_path,support_path,support_receipt_path=None) returns
   draft method payload without headers/keys/values.
2. freeze_payload(payload,frozen-selection-method.yaml) after READY/suspension.
3. prepare(frozen_selection) writes byte cuts/config/receipt or unsupported;
   no job is launched.
4. bind_comparison(frozen_selection) returns key/schema-bound payload; root
   freezes it before command/value access.
5. verify_comparison(frozen_comparison,before_replay=True), then root runs the
   receipt command and writes execution-receipt.yaml.
6. check(frozen_comparison) writes fresh validation.yaml; no replay/fit.
