# Component-specific native producer integrity method

This independent integrity method is evaluated and frozen before a fresh V3
baseline or H10 FE comparison. It changes no sampling, label, split, CatBoost
formulation, metric or predictive acceptance rule. Historical NativeGateV1
failed its original whole-producer correspondence requirement; that failure and
its hashed receipt remain unchanged. The new method can pass only the narrower,
explicit component lineage claims below.

Two fixed 20260203 controls, 2609 and 2337, distinguish three producers. The
historical original binary is SHA256 `2374c82c3fd60f02e636fd2eec54329b60324228b8febefc29e463b3ae57ff60`.
Its newly regenerated original-config parquet must be byte-identical to the
original published parent file. Preserved pre-H10 binary
`69b5123804b321e9f09f2c3b42511a232f2dacbc519688754282fc63c73f6c34`
and the frozen H10 binary must also produce byte-identical control files without
H10 instantiated. Neither claim is substituted for the other.

Historical original versus preserved current schema and row order must agree.
Every native key, raw float64 OriginMidPrice, original endpoint and sticky-return
label, and all other feature/metadata fields must retain exact bits and Arrow
validity. Complete identity is `(day, symbol, SampleTime, SampleBookTime,
SampleBookSeq)`; day/symbol bind the declared single-symbol/day partition, and the
three actual native columns remain nonnull int64. No cast, sorting, recoding,
as-of join, tolerance, forward fill or special-value translation is performed.

Only these documented differences are accounted for:

- FlowResponseSurprise: flow_surprise0..2, response_innovation0..2 and
  response_coupling0..2, each `.0` slot. Only original NaN to current negative
  infinity is permitted. Finite-bit changes and every other state transition
  reject. The original producer incorrectly lost unfinished warmup on stale or
  ambiguous observations; its corrected source SHA is
  `7cc10d17d0c47d8f845b5fcf9bfbe99040dfae069931cd711c16fc5dc2ad97cc`.
- CrossReturnContext: target_return_z0..5, residual_return_z0..5,
  target_sigma_bps and peer_sigma_ratio, each `.0` slot. The earlier documented
  halt-boundary correction can change these normalization values and states.
  Each changed field, changed-row count and state transition is reported and
  bound to the historical correction receipt. The exact changed-field set,
  changed-row/finite-row/state-change counts and origin count must match its
  `reference_ten_second` evidence. Other CrossReturnContext fields are not exempt.

The checked-in exact-name allowlists cover 9 and 14 columns, never whole-family
wildcards. The historical correction evidence is
`runs/fe_origin_20260930/warmup-state-parity-1/results.yaml` and
`experiments/fe_origin_20260930/SOURCE_AUDIT.md:103`. Historical research keeps its
original values and engine; this method neither rewrites those inputs nor makes
the corrected producer retroactively part of an earlier frozen comparison.

The four fixed TRAIN pilot joins, 20260119/20 by 2308/2317, must preserve every
original five-part origin in order and the raw float64 mid bits. H10 feature-only
files contain no regenerated native labels. Both ordered/reset instances must
retain integer-valued exchange time no later than receive availability, receive
availability no later than origin time, matching
zero/unobserved clock states, and exact phase/missing-state meanings: warmup
negative infinity, inactive positive zero, censored/stale NaN, active finite.
Diagnostics never become model covariates. Full V3 material still requires the
same exact joins on every actual model row; these four pilot controls do not
claim regeneration of the complete historical dataset.

The manifest is `native-h10-ordered-reset/component-lineage-manifest.yaml`.
`method_contract` binds frozen YAML bytes, its separately hashed `.identity`
anchor and the declared identity. The contract's `identity` must equal
`astra.io.digest` of its body excluding that field, the adjacent `.identity`
contents, and the manifest's declared identity. Every contract source/input is
verified and merged into the output under the same scope; the executing validator
must match the frozen validator source SHA. The separately frozen reproduction plan must
precede original-engine replay. Each original/current/H10 producer binds its
immutable binary, runtime dependencies and source/build receipts; every native
artifact, config/raw dependency receipt and historical failure/correction receipt
also has an exact SHA256 binding. Source paths refer to immutable snapshots.
The validator verifies bindings before comparing artifacts and records all
bound inputs and sources. A missing, changed or mismatched binding rejects.
The original-binary reproduction uses the explicitly frozen current runtime
dependencies. Its byte equality does not assert historical runtime equality;
`historical_runtime_identity_claimed` is always false.
The required `manifest_payload_identity` equals `astra.io.digest` of the manifest
excluding only its `method_contract` anchors. It is frozen inside the method
body before those anchors are filled, binding every declared producer role,
artifact path, cell and receipt without a circular hash. A manifest whose roles
or bindings differ rejects before native artifacts are compared.

The isolated implementation is `astra/representation_provenance/validator.py`.
Run it from AstraResearch with `python3.13 -m
astra.representation_provenance.validator --manifest <frozen manifest> --output
<fresh producer-lineage-validation.yaml>`. It never builds, replays, fits,
regenerates labels or overwrites an existing receipt. Synthetic contract tests
must reject unaccounted differences, illegal FlowResponse transitions, float
key/mid recoding, wrong origins, future availability and false hash/identity
anchors before production acceptance.

The output schema is `native-component-lineage-validation-v2`, with `passed`,
`baseline_parent_reproduced`, `current_control_invariant`, component and candidate
join decisions, preserved NativeGateV1 failure, producer binary identities,
`method_contract_identity`, and hashed sources/inputs. A pass establishes these
bounded component integrity facts. It does not establish whole-producer byte
parity with historical material or any predictive improvement.
