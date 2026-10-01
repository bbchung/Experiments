# H16 next comparison design — unadopted draft

Source-only design review, 2026-10-01. This document freezes no methodology,
authorizes no fit and changes no existing source, profile, sample, label or
acceptance rule. It reads the small V5 report and native declarations/formulas;
it reads no feature matrix, score array, label values or raw market tape.

The raw-routing H16 pilot's original-MID mismatch is an integrity failure, not a
predictive rejection. A new explicitly versioned producer must first preserve
the original TwseFilter/CurrentBook keys and MID bits while separately exposing
the raw observations. The comparison below remains conditional on that proof
and an independently frozen native support check. The failed producer and its
contracts remain immutable.

## Five primary 300s arms

| Arm | Representation | Purpose |
| --- | --- | --- |
| B | Fresh original legacy control | Scientific comparator under the unchanged judge |
| Bcat | B plus the two H16 categories | Observed regime/cycle context without numeric ledger operands |
| Bfull | B plus all seven H16 fields | Incremental ledger information with legacy context retained |
| C18 | Fresh normalized native C18 | Same-run compact-context control |
| C18full | C18 plus all seven H16 fields | Only eligible complete normalized representation |

All five share the exact V3 judge fields, original 30s origins, all-four-horizon
maturity cohort, labels, split/universe, class/recency weights, CatBoost loss and
parameters, early-stopping procedure, metrics, calibration nomination timing,
development procedure, seen/held guards and acceptance functions. No feature
availability or auction regime selects rows. Categorical and numeric operands
must come from the same native producer at the same origins.

The five arms answer different questions, without a feature-family grid:

- Bcat minus B tests whether explicit observed regime/cycle categories add
  information to the legacy control.
- Bfull minus Bcat tests the five numeric ledger operands conditional on those
  exact categories and retained legacy context.
- Bfull minus B tests the full source block without discarded-context confounding.
- C18full minus C18 tests transfer of the full block into normalized context.
- C18full minus B tests the only eligible complete representation.

There is no C18cat arm. Consequently, a C18full gain cannot be assigned
specifically to its numeric ledger rather than categories or their interaction.
The numeric-versus-category attribution is established only in the B context.
This is a declared limitation, not a reason to add another default arm. Nor does
Bfull success prove that the raw-scale legacy universe is suitable for pooled
deployment. B, Bcat and Bfull are diagnostic controls and cannot become nominees.

Nomination must remain a persisted calibration decision made before development
diagnostics. Only C18full is eligible. Development diagnostics for every arm,
including when calibration nominates none, must be authorized prospectively.
The C18full-versus-C18 contrast uses the existing judge and paired procedure;
the new profile must explicitly say whether it is descriptive or an additional
required acceptance contrast. Silently making it an admission requirement would
change the acceptance contract. No raw-arm winner or development-only winner
can substitute for the declared nominee.

## What V5 justifies, and what it does not

V5 rejected C15/C18. Development C18 lost 8.38% large-move AP versus B, and
the paired magnitude/joint intervals were wholly negative. The small pooled
direction increase coexisted with a lower equal-symbol/day direction AUC. Thus
H16 should be tested for magnitude opportunity as well as proposal direction;
a directional-only gain cannot establish a complete representation. A numeric
ledger magnitude contribution can be scientifically useful even if this fixed
complete-representation judge rejects it, but it creates no alternate admission.

V5 TRAIN model usage points to possible omitted context, not causal attribution
or a request to restore the largest-importance families. Static neighbors cover:

| Native neighbor | Existing information | Boundary of the H16 claim |
| --- | --- | --- |
| MaturedPathRegime | Fully elapsed sampled sticky-path peak, path length, endpoint retention and dispersion | A path summary is not a recurring actual-match/proposal relation |
| MaturedBarrierOutcome | Matured bid/ask first-barrier outcomes and excursion/timing history over book-flip cohorts | Merely changing its barrier to five or extending its horizon is not a new source |
| BreakRetention | Recently broken sticky extremes, delayed retention and refill | Generic break persistence is already represented |
| LiquidityTransmissionState | Capacity-relative execution, survival/reload, repeated defense and exhaustion | Generic work-to-price transmission is already represented |
| CauseConditionedQueueResponse | Cause-specific in-flight work, recovery/escape/timeout outcomes | A generic response or competing-outcome ledger is already represented |
| PriceMemoryState | Executed-volume price memory and nearby path pressure | A generic acceptance histogram is already represented |
| AuctionCarry | Trial/actual auction primitives and opening/VI carry | H16 novelty is its recurring causal cycle ownership/publication, not new primitive inputs |

No existing neighbor's importance or name establishes whether H16 adds signal.

## Compute and interpretation boundaries

Five serial fits are a bounded attribution experiment, not HPO. V5 observed
22.674s for B and roughly 5.5s for each compact fit; these numbers establish only
local prior cost, not a guaranteed cost for augmented matrices. The larger risk
is repeated matrix allocation and source hashing. Reuse the original immutable
arrays, one small canonical H16 block, one common cohort proof and cached
source/input hashes. Construct at most one required contiguous augmented matrix
per fitting stage; do not persist three copies of B or retain redundant gathers.
One root-owned GPU process runs at a time.

Keep the exact seen/held groups. The Jan19 recurring-auction support examples
3481/2344/2337 are seen symbols; they do not prove held-symbol auction support.
Descriptive regime exposure should use causal native state at each origin,
common across all arms, with unknown/stale/partial rows retained. It must not
classify a whole day from later observations or adopt the separate, unadopted
disposition-metadata method. Report counts, day/episode support and undefined
metrics explicitly; do not delete an unfavorable group or turn a sparse group
into an alternate acceptance cohort.

H16's residual MID may already contain the trial-price direction. Therefore
Bcat gain with no Bfull-minus-Bcat gain supports an observation-context
interpretation rather than a numeric-ledger claim. A Bfull gain that fails to
transfer to C18full suggests missing conditioning, interactions or scale proxies;
it does not nominate Bfull. A C18full gain on pooled direction alone, without
the fixed within-group/magnitude/paired requirements, remains a rejection.

## Horizon hierarchy

Primary 300s is appropriate for the proposed recurring-cycle information clock,
but a historical receive interval is not a matching deadline. No countdown,
modulo phase or future cycle interval is manufactured. Two secondary 60s fits
(fresh B versus the selected C18full) only after primary calibration nomination
are a reasonable predeclared compute hierarchy. They must train on the original
60s pure-MID labels under the same CatBoost procedure and original common cohort;
scoring a 300s model against 60s labels would answer a different question.

Those secondary fits are descriptive, with no candidate reselection, new
parameters or alternate primary admission. This hierarchy cannot diagnose
short-cycle-only usefulness when primary nomination is null, and it says
nothing about 120/180s effectiveness merely because those labels determine
cohort maturity. If unconditional 60s failure diagnosis is required, that budget
and interpretation must be declared before the comparison freeze, not added
after seeing a failed primary result.

## Distinct fallback source question

A stronger alternative to appending generic path/refill variants is **raw
market-order ownership versus priced extreme-limit liquidity**. This is a
source-contract hypothesis, not an observed market-data or predictive result.
TwseFilter's header defines a zero-price touch with volume as a market order.
Its normalization rewrites that price to the contract limit and merges a
following equal-limit priced level. A raw market quantity 10 plus priced-limit
quantity 20 can therefore become the same normalized touch as a single priced
limit quantity 30. Their different order-type ownership is not recoverable by
longer windows over the normalized book.

A future compact native representation could preserve source-known market-order
mass fractions and signed imbalance separately from priced liquidity. Use exact
integer source identity/zero decisions and same-symbol observed quantity shares,
not raw pooled quantities or an epsilon inverse. Unknown sides, absent reports
and reported zero mass remain distinct. Do not claim a clearing curve, hidden
supply, urgency or exogeneity merely from the zero-price field.

Before implementing it, reject or retain this hypothesis with a bounded
TRAIN-only source-support study and native counterexamples: identical normalized
books but different raw order ownership; exact zero versus absent/unknown;
source-preserving raw capture; and sufficient original-origin support. Audit
AuctionCarry and actual source field semantics first. If the source distinction
is absent, already exported, or fails support, reject it without a new feature
or symbol substitution. A surviving representation must face a fresh baseline
under a separately frozen, unchanged FE judge; support is not predictive proof.

## Source references

- `RESULTS_V5.md`, `profile-v5.yaml`, and the unchanged representation-research evaluation functions.
- `src/oms/modules/feature/microstructure/event_path/{matured_path_regime,matured_barrier_outcome,break_retention}/`.
- `src/oms/modules/feature/microstructure/liquidity/liquidity_transmission_state/` and `microstructure/queue/response_lifecycle/`.
- `src/oms/modules/feature/microstructure/price_memory/price_memory_state.*`.
- `src/oms/modules/feature/market_state/exchange/auction_carry/`.
- `src/oms/modules/tw/twse_filter/twse_filter.{h,cpp}`.
