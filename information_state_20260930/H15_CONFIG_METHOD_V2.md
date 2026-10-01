# H15 native configuration correction V2

This is a prospective producer configuration repair. H15 V1 failed while
constructing `TwseFilter` because `StatusFilter: ''` is invalid expression
grammar. No H15 feature outputs were produced or decoded. The V1 frozen method,
profile, preparation, source capsule, startup log and failure manifest remain
immutable. This failure provides no mechanism support or predictive evidence.

V2 changes both leg filters and `PeerTradeInformation` to the explicit scalar
expression `TRIAL || !TRIAL`. The registered native status parser implements
`||` as a uint16 truth-table OR and `!` as its uint16 complement across sixteen
status classes. Thus `T | uint16(~T) == 0xffff` accepts every class. The preserved
compiled parser source and native status-filter tests are mandatory evidence;
Python string-type validation alone does not establish valid parser grammar.
The independent helper tests enforce this exact expression and reject an empty
expression or a restrictive replacement. Successful native startup will only
be claimed after root executes the new jobs.

The scientific hypothesis, eleven Alpha fields, twenty-seven Info fields,
physical arithmetic, status/clock/state ownership, target/peer identities,
sixteen-carrier order, two TRAIN dates, original native keys and mid bits,
10-second writers, 30-second support origins, event writer and every support
gate are identical to the frozen V1 H15 native plan. Labels remain absent;
there is no model fit, FE comparison or OOS claim in this native support study.
Missing fixed inputs retain the original skip rules and receive no replacement.

The binary SHA256 remains
`613260125eb524b3e2c5af77a8621083b28cacfedc488d421c06e7709961fd6b`.
V2 references the existing V1 binary, 1648 compiled source copies and the shared
51-library runtime without copying or rebuilding them. Every V1 compiled/live
source and the live built binary must still match before and after preparation
and at preflight. Any drift requires a separate study version.

V1 already executed the registration control successfully. Its four complete
H13 Parquet artifacts are byte-identical to the original H13 controls. V2 binds
the ordinary successful execution receipt, old command/config, log, four output
hashes and required H13 output hashes. It reuses those artifacts; the control
has `requires_replay: false`, an empty executable command and a separately
preserved `reused_command`. It cannot appear in the new execution order. The
V1 startup failure manifest, failed execution receipt, native log and run status
are also mandatory inputs and prove this is a configuration-only continuation.

Root alone invokes the new helper's `--prepare`, which generates the V2 profile
and two fresh daily work directories. Root then freezes canonical method schema
`h15-native-producer-method-v2` and its identity anchor, including all shared
compiled sources, the new helper and plan, unchanged native plan, V1 method and
identity anchor, new profile/preparation/capsule, grammar proof, raw identities,
original native keys, reused registration artifacts, runtime and failure proof.
Preflight reconstructs mandatory dependencies rather than trusting an omitted
manifest list, verifies the exact daily command/config/export/sampling recipe,
rejects registration relaunch and existing new outputs/status, and accepts only
the two fixed daily job keys. Root executes them serially only after freeze.

The new evaluator binds the new producer identity before reading new native
outputs. Native support counts, sampling diagnostics and rejection criteria
remain prospective and unchanged. This document does not adopt a predictive
representation or alter any frozen FE evaluation judge.
