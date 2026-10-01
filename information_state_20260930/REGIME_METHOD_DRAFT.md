# Ex-ante exchange-regime methodology research draft

Unadopted source-methodology research. This document and the isolated
`astra/representation_regime_method` parser do not change any frozen profile,
sampling, labels, judge, native source, feature representation or symbol set.
The original fixed sixteen-symbol H15 mapping remains rejected. Positive source
support in other cells is not predictive evidence or a reason to delete failures.

The information problem is specific: H15 demands an ordinary continuous prior
book and subsequent execution-confirmed price discovery. Indicative snapshots
of a scheduled periodic call auction are a different observation mechanism.
Finite two-sided CurrentBook mids do not establish continuous price discovery.
An appropriate common cohort, or separate regime-specific representation, must
be studied and frozen independently before a fresh baseline/FE comparison.

## Bounded source evidence

Local metadata directory: `/mnt/data0/info/tse/stock/disposition/`.
CSV fields are `編號,公布日期,證券代號,證券名稱,累計,處置條件,處置起迄時間,
處置措施,處置內容,備註`. Publication uses ROC YYYMMDD; active spans use
ROC `YYY/MM/DD～YYY/MM/DD`. The text explicitly describes approximately one
matching operation every five minutes. It also mentions conditional extension
after suspension or calendar changes: a printed expiry is not an unconditional
future promise of continuous trading.

| Symbol | Announcement | Inclusive active span | Explicit matching interval |
| --- | --- | --- | --- |
| 3481 | 2026-01-06 | 2026-01-07 through 2026-01-20 | about five minutes |
| 2344 | 2026-01-08 | 2026-01-09 through 2026-01-26 | about five minutes |
| 2337 | 2026-01-09 | 2026-01-12 through 2026-01-23 | about five minutes |

All three notices are already present in the prior-calendar-slot snapshots
used for the fixed TRAIN dates: January 16 for January 19 and January 19 for
January 20. Both snapshots have SHA256
`f34bc373208ed6323b5d13de10cd75e1eb705802a634f37e749a3ec853d17ce7`.
January 20 metadata was inspected only to cross-check notice stability; it is
not an admissible pre-session source for January 20 classification.

The January 6 snapshot (SHA256
`d9269731d759b1d19bc6bb913f413e954e4089b10d1c48b674882be2f7309452`)
does not contain 3481, although a later file records publication January 6 and
activation January 7. January 9 (SHA256
`7aaeac7dd8c404e48ddd30f55d540e15d7fda935ff6c8fe578b623e025a32978`)
does not yet contain 2337 despite its recorded January 9 announcement. This
falsifies the inference that a prior-day file is an exhaustive post-close list
of all next-day announced schedules. Its filename provides no capture time.
Absence is therefore UNKNOWN, never continuous admission. The draft classifier
has no option to override this missing completeness proof with a boolean.

Root independently read existing, already hash-bound January 19 native prefix
spools: two legal positive-queue Book records per affected target near 09:10
carry StatusMask=3 (TRIAL|AUCTION), with no continuous-status book found in its
bounded spool search. Those findings are consistent with the positive prior
notices, not a claim that a complete January 20 tape was newly decoded.
Witness source closure is `native-h15-receive-prefix/preparation-receipt.yaml`:

| Existing spool | SHA256 | Bytes |
| --- | --- | --- |
| `spool/3481.bin` | `39a08f7380e5e0a2b9e031f2fa83b30e20a7d45f11511ff428b2ccf0a0599104` | 921088 |
| `spool/2344.bin` | `8de9f8bd92770f4d7b9020df91b7c66c83dd235d2e5a03b38eba36a8515275a6` | 920320 |
| `spool/2337.bin` | `09194e875fc60deee1ba5bec607500a406187ba9178c024480dd34fccd34be1b` | 912400 |

Existing mapping exports show healthy peer-source counts but zero closed target
availability and rising hard-censor counters in these cells. Native
`PeerTradeInformation::normalized_book` rejects noncontinuous status before
price-grid checks, and `CurrentBook::on_book` publishes raw mids/ticks without
that status restriction. These source semantics explain the failure. There is
no evidenced grid/float32 or trade-permission bug to repair.

## Proposed reproducible decision boundary

1. Bind the original universe, declared calendar and role days before any
   label/model work. Choose exactly the previous declared trading-calendar slot,
   within the 2026 research floor. Never search for a nearest existing file.
2. Bind each required CSV path/SHA/schema. Missing required prior metadata gives
   UNKNOWN/skip without fallback, replacement dates or assumed normality.
   Reject malformed dates, future-publication rows in an alleged prior snapshot,
   duplicate semantic notices, ambiguous overlapping notices and changed files.
3. A strictly prior published notice with an inclusive active span and explicit
   auction interval provides positive batch-auction mechanism evidence. An
   active disposition with no unambiguous interval remains UNKNOWN. A printed
   expired span with unresolved conditional-extension text remains UNKNOWN.
4. No known notice is not yet proof of continuous eligibility. A later
   independent source study must establish exhaustive announced-schedule and
   amendment coverage plus capture availability before allowing that complement.
   The present metadata implementation deliberately cannot produce a positive
   continuous-eligibility result.
5. Regime classification is pure signal: it consumes announcement mechanism,
   dates and source availability, not day_trade permission, short-sale status,
   costs, labels, support success or model scores. BasicInfo No cannot substitute
   for auction evidence. 2409 limit lock is separate current feature availability
   inside a continuous schedule, not a static batch-eligibility criterion.

Schema/date parsing and boundary rejection can be validated cheaply with unit
fixtures and these already inspected metadata sources. Native-status concordance
is descriptive; an isolated noncontinuous book does not prove an all-day batch
regime, and a daily auction schedule can include actual matching observations.
The helper never turns such witnesses into predictive or methodology admission.

## Source coverage and the next independent study

A stat-only inventory of required prior snapshots under the existing V3 role
calendar found TRAIN102/102, ES9/9, calibration20/20 and development24/30. Six
development dates have missing prior metadata. This is metadata availability
only: no feature matrices, labels, models or extra market tapes were read. It
does not validate complete negative announcements or authorize a smaller cohort.

Existing `astra/signal/regime.py` instead calculates same-day whole-session
continuous-book fractions and filters origins/endpoints. Its day-wide exclusion
uses later observations relative to early origins and must not silently replace
the frozen judge or be represented as ex-ante eligibility.

The most direct complementary source study is an as-of-origin observation
contract. Existing CurrentBook has no raw StatusMask Info export. A future small
native diagnostic producer could expose last received book status/identity and
its availability, allowing an identical causal continuous-origin sampling rule
for B and every candidate. H15 K invalid conflates status, grid, visibility and
clock failures; selecting by that feature would be an unsuitable regime proxy.
No native implementation or new producer is authorized by this draft.

Before any FE comparison, a new methodology must freeze source/cohort rules and
prove exact receive-prefix availability, missing-source behavior, metadata-date
boundaries and stable common original observation identities on a bounded TRAIN
study. The present negative-completeness counterexamples must stay visible. If
this method is adopted after that independent validation, restart fresh B and
all candidates under the same new contract; retain old mapping FAIL unchanged.
Alternatively study batch-auction indicative-to-matching information separately
under an independently frozen label/sampling interpretation, retaining pure mid
price changes and semantic cross-symbol physical scales. No OOS success filter,
whole-day future eligibility, peer substitution or threshold search is permitted.
