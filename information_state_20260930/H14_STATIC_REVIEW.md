# H14 static review: reject a mid-target traversal-cost interpretation

Status: source-based proposal review only. No features were implemented, no
dataset or model results were read for this review, and no predictive admission
is claimed. This document is outside the frozen V3 comparison closure.

## Decisive semantic limits

With a one-tick spread and contiguous one-tick quote levels in one tick band,
the five displayed bid/ask levels end at mid minus/plus 4.5 ticks. The price at
mid minus/plus five ticks is outside the displayed terminal quote. There is no
observed terminal book at that target from which to infer a completed walk.

This fact alone does not prove missing quantity *inside* a fixed five-tick
price cutoff. Under the ordered legal-quote contract, the next legal level after
the contiguous fifth level is at 5.5 ticks, outside that cutoff. The displayed
quantity sum inside the cutoff can therefore be complete even though the
terminal price and its future quote state are not observed. A general producer
must establish this coverage from actual legal prices and the feed's observable
book contract, rather than assume it from level count or manufacture a tail.

More fundamentally, consuming ask quantity in a mid-plus-five price corridor
does not imply a plus-five mid move. For a locally linear ladder, a mid move of
five ticks requires the two touch displacements to sum to ten ticks. The bid
can remain fixed, follow, retreat, or change independently; replenishment and
quote withdrawal can also renew either touch before a static walk completes.
Across tick bands, arithmetic price mid and the mean of quote tick coordinates
must additionally remain distinct. No observed-book quantity formula establishes
that joint future quote mechanism.

An empty corridor in a wide spread means no currently displayed quote lies in
that price interval. It does not mean zero resistance to a mid move. Unknown
book state, observed empty price intervals, and an actual zero quantity must
not share a manufactured zero, epsilon denominator, or inverse-cost meaning.

## Honest residual interpretation and overlap

A retained observable would be **standing displayed quantity in a fixed price
corridor**, with explicit complete/partial/unknown coverage. Under partial
coverage its sum is a lower bound on nonnegative displayed quantity, not a lower
bound on execution cost, movement time, or future liquidity. A bounded ratio
against a positive, causally available prior-volume unit would mean relative
standing mass. It would still require empirical validation and could not be
named a five-tick mid-traversal probability or capacity.

Existing neighbors already cover much of this residual information:

- `VisibleBookImpact` walks 0.25/0.5/1.0 times the smaller displayed side's
  quantity and exports terminal/VWAP tick geometry.
- `BarrierWorkState` compares normalized trade/touch/away work with standing
  capacity at level depths 1/3/5; unpriced work retains work since a sticky move.
- `LiquidityTransmissionState` combines depth-fraction shock geometry with
  normalized attacks, defense, reload and exhaustion.
- `PriceMemoryState` already keeps session and recent price-volume histograms;
  its six raw log-volume Alpha outputs require a new semantic volume reference.

Exact sources are respectively
`src/oms/modules/feature/microstructure/execution/visible_book_impact/visible_book_impact.cpp`,
`src/oms/modules/feature/microstructure/liquidity/barrier_work_state/barrier_work_state.cpp`,
`src/oms/modules/feature/microstructure/liquidity/liquidity_transmission_state/liquidity_transmission_state.cpp`,
and `src/oms/modules/feature/microstructure/price_memory/price_memory_state.cpp`.

## Decision

Reject H14 as a target-traversal-cost mechanism before implementation. Merely
changing the existing walk's cutoff to five ticks, or dividing its quantity by
past volume, does not establish a fundamentally new causal information source.
The compact twenty-field context recipe remains an untested replacement
proposal; it is not admitted by this static review.

A price-visit acceptance mechanism is a stronger next question if it observes
information absent from these neighbors: which previously established quote
became available first, whether a later strictly known execution accepted that
new quote, and how the opposite quote responded, all at received availability.
It must preserve visit identity and ambiguity rather than infer participant
identity or summarize an unordered volume histogram. H13's separately declared
TRAIN-only support gate takes priority. Its failure must not be rescued through
a longer window, replacement symbols, or a relabeled corridor interpretation.
