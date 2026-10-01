# Native temporal validation method: independent result

The independent method is frozen as
`182d8d530e8f75fdb1861b6ee6a3a9b99971763fbcbdf5bd7855e9a243304255`
at `runs/information_state_20260930/temporal-study/frozen-method.yaml`.
Its canonical identity, separate anchor, mandatory evidence/source closure and
reconstructed validation are verified before any V3 FE contract or fit.
The freeze binds 1,647 sources and 14 inputs, including all 1,637 original
compiled source copies and the independent native test source/binary/logs.

The complete already-produced native population contains 2,576 symbol-day
cells and 3,540,544 observations per chronology variant. Both variants have
zero receive availability after the sample origin. Vendor exchange is ahead
of availability in 30,802 observations across 126 cells, and ahead of the
sample origin in 5,268 observations across 102 cells. These are descriptive
cross-domain offsets; neither count licenses future receive information.
The actual raw 2026-02-11/2481 BOOK closing pair confirms availability 42,128
microseconds before the sampled origin, despite an exchange timestamp 4,842
microseconds ahead of receive. No marketdata timestamps or rows were changed.

The separate native metamorphic test shifts only the unit fixture's exchange
clock by plus/minus one hour. All seven numeric AlphaFactor bits and phase
match for ordered and reset variants, including staleness, pending clusters,
censoring and rearming. All 24 native unit cases passed. The original H10
producer binary remains SHA256
`8678aa7fd4cac798909211164284423b5e12c0e32ca021367a05962a5e07b0bd`.

All 17 Python method tests passed. They cover exact receive boundaries,
native clock dtypes, strict `<2**53`, fractional/nonfinite/negative/negative-zero
states, paired zero and observed-positive semantics, raw witness scope,
unchanged producer binding, freshly canonicalized closure omissions, validation
reconstruction and profile/source/anchor drift. Missing mandatory evidence is
rejected even if a modified manifest is given a new self-identity.

This admits the independently justified temporal integrity rule: receive
availability must be at or before receive-domain origin; no ordering between
vendor exchange and receive clocks is assumed. It changes no AlphaFactor,
label, sample, split, training, predictive metric or FE acceptance rule.
The failed earlier V3 preparation and old frozen provenance receipts remain
recorded. There are no CatBoost fits, model scores, predictive gains or OOS
claims in this methodology result.
