# Proposed H2 follow-up: quote translation versus spread reshaping

Version 1. Status: root approved, pending source/input freeze and independent tests before computation. This is a new explanatory family on exposed research validation, not an H2 rescue, candidate, policy search, or promotion gate. H2 R remains unsupported in August and D remains economically rejected. No new result in this family changes those decisions.

## Question and competing explanations

Does the positive D midpoint diagnostic represent movement in both executable touches, or mostly one-sided/offsetting quote changes that reshape the spread? A secondary question is whether the information retains the same character at the already-frozen 250 ms delay and 1/5/30-second endpoints. Midpoint movement alone cannot distinguish these explanations.

H-translation predicts positive signed bid and ask movements together, with positive shared touch movement that persists across the fixed delays/horizons. H-reshape predicts that positive midpoint accounting comes mainly from one-sided/offsetting changes and the residual beyond shared touch movement, often associated with entry/exit spread changes. Mixed mechanisms and unresolved evidence are permitted conclusions.

## Frozen population and inputs

Use CDF only, both original R and D views, every planned July and August-through-26 day, and the existing H2 feature-only union and five-second spacing. Do not reconstruct or optimize intents. Hash the H2 registration, amendment, all-intents, all-events, daily, coverage, and summary inputs before computing. Use the original active flag and sign; inactive rows remain explicitly inactive and unknown active rows remain unknown.

The primary decomposition is the fixed 5-second horizon at both 50 and 250 ms entry delays. The already-exported 1- and 30-second outcomes are fixed checkpoint-persistence diagnostics, never alternative winners. Every horizon/delay retains its original known/unknown population and reason. Report the 50/250 ms common-known 5-second population separately, and a second explicitly labeled common-known 1/5/30-second population within each delay for checkpoint comparisons. Neither intersection replaces any original coverage denominator. Agreement at these endpoints does not establish uninterrupted movement throughout the interval.

Use saved target entry/exit bid, ask, receipt time, age and quantity fields. These are the exact quotes used by H2, preserving its strict-prior arrival, validity, freshness, depth and path-episode rules. No new quote matching, endpoint, cost, target, feature, model, universe or calendar definition is introduced. Do not assign target outcomes to unknown rows.

## Deterministic accounting

For each known active event let s be its frozen direction (+1 long, -1 short), m0 the delayed entry midpoint, and K=10000/m0. Define:

- B = s * (exit_bid - entry_bid) * K: signed bid-to-bid movement.
- A = s * (exit_ask - entry_ask) * K: signed ask-to-ask movement.
- M = (B+A)/2: signed midpoint movement.
- W0 = (entry_ask-entry_bid)*K and Wh = (exit_ask-exit_bid)*K; report both and Wh-W0.
- Paid crossing = (W0+Wh)/2; aggressive gross = M-paid crossing. Check this exact identity against H2 gross bps for both directions.
- Shared touch movement T = sign(B)*min(abs(B),abs(A)) when B and A have the same nonzero sign; otherwise T=0. Nonparallel quote-change accounting Q=M-T. Check M=T+Q exactly within floating-point arithmetic. This is a declared accounting convention, not an identified fundamental-value or causal spread component.

Join the unchanged native `OriginMidPrice` from the saved intent by exact day/product/SampleTime. Separately report signed origin-to-delayed-entry midpoint movement P = s*(entry_mid-OriginMidPrice)*K, and check P+M against the origin-to-exit change using the same entry-mid denominator. Preserve unavailable native origin values as unknown annotations rather than changing economic coverage. This distinguishes movement consumed before entry from movement after arrival. There is no saved origin bid/ask-price decomposition in this slice, and no Python tick-ladder reconstruction is introduced.

Use exact zero on the saved quote prices; do not introduce a fitted epsilon. Classify every known active event into exactly one of: both touches agree with the signal (B>0,A>0), both oppose (B<0,A<0), only one moves with the signal, only one moves against it, touches offset (opposite nonzero signs), or unchanged. Preserve all classes, including adverse and empty classes.

Report event counts, original known/active fractions, equal-day means of P/B/A/M/T/Q/W0/Wh/spread change/gross/net, and cash sums for each arm/month/delay/horizon. Zero-active days retain zero policy accounting; an active day with no known outcome remains unknown. Also show contribution to the overall mean using the same whole-arm denominator, so rare classes cannot look important merely through conditional means. Conditional class means are descriptive only. Retain full daily values and positive/negative-day counts and leave-best-day-out means/concentration. No significance or multiplicity-adjusted confirmation claim is made from this explanatory family.

For the primary 5-second D diagnostic at each delay, retain 4,000 paired circular three-observed-trading-day bootstrap draws within each month, seed 1729, for daily B/A/M/T/Q and hypothetical passive net means. Report the 2.5% and 97.5% quantiles for each month and for the pooled within-month draws. Use identical resampled day indices for all metrics. Unknown whole days are excluded with their original calendar-coverage record; an unresolved active day blocks that metric's interval rather than becoming zero. These are descriptive intervals, not a new promotion gate.

Report the complete decomposition within the predeclared H2 narrow/wide origin-spread cells and quiet-target indicator, with all cells retained. Do not introduce quantile searches or favorable subsets.

## Causal receive-order context

The existing native flip traces support a strictly pre-origin context label: last reference flip later than last target flip; last target later; equal receive timestamp; one or both absent. Preserve ties and unavailable history explicitly. Describe this only as received flip order, never exchange-time leadership or which market supplied information. Price-flip traces omit size-only updates. Use the native timestamp keys already audited; do not use future next-flip times to assign origin classes. No reference-price outcome is imputed from a flip key. Joint stock endpoint movement is outside this bounded diagnostic because it was not saved on the exact target endpoint population; adding it requires a separately declared source/coverage extension.

## Mandatory hypothetical passive-entry accounting

Calculate one fixed descriptive benchmark from the same known quotes: instantaneously buy at entry bid and exit at the future bid, or instantaneously sell at entry ask and exit at the future ask. Use exactly 2,000 shares per contract and charge 50 TWD commission per side plus 0.00002 times each hypothetical transaction's notional. This is B before fees for long and A before fees for short. Report both months, views, delays and horizons, and its difference from actual H2 aggressive-entry accounting.

This benchmark assumes an immediate passive fill with no queue, adverse selection, missed fill, information delay or price movement while waiting. It is not a fill simulation, feasible strategy, upper bound for a selective policy, or evidence supporting production. Negative benchmark results weaken that particular instantaneous-entry story; positive results establish only a reason to investigate actual fills under a new frozen execution design.

## Evidence and decisions

No optimized winner threshold or candidate ranking is used. Economic zero and the original two months/delays/horizons organize interpretation.

- Translation evidence strengthens when shared touch movement T is positive in both validation months at both delays and the same events retain positive touch movements at the fixed 1/5/30-second checkpoints. It weakens if midpoint gains coexist with flat/adverse shared touch movement or disappear/reverse in the fixed paired checkpoint view. No endpoint statistic identifies a latent fundamental-value change or proves uninterrupted persistence.
- Reshaping evidence strengthens when positive M is accounted for by Q/one-sided/offset classes while T is flat or adverse, and the full origin-spread decomposition supports that explanation. It weakens when both touches shift persistently with little residual reshaping. Do not force an exclusive conclusion when both occur.
- If translation persists but remains too small relative to paid crossing, the information is economically insufficient for H2's aggressive round trip. An execution investigation is justified only as a new question, with fill probability and adverse selection central.
- If reshaping dominates or useful translation is absent, deprioritize model/HPO and simple sampler changes for this target; next consider whether an executable-touch target can distinguish true adjustment from midpoint noise before training anything.
- Unknown support, calendar omissions, month disagreement or strong dependence on a few days limit the conclusion; no exclusions or reinterpretation as final OOS are allowed.

Retain the report, per-event decomposition, full daily/monthly/class tables, original coverage, paired-intersection coverage and input hashes. No canonical workflow or candidate is modified by this diagnostic.
