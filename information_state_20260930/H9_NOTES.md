# H9: sampled-price renewal hurdle and marked episodes

V1's train price history has a zero median absolute 60-second move for fifteen
of sixteen stocks. Its magnitude/occupation replacements failed the original
calibration nomination. This motivates a different stochastic representation:
price mobility consists of an arrival process and marks conditional on arrival.
An unconditional median suppresses the probability mass away from zero. Arrival
and conditional step direction/size can carry different information about a
five-tick endpoint. These are hypotheses, not evidence of a gain.

`astra.representation_research_v2.renewal.build(frame, profile)` reads only day,
symbol, native integer-microsecond SampleTime and float64 OriginMidPrice. Its
twelve fields separate observed renewal, same-clock renewal probability,
age-conditioned renewal probability, dwell survival, current-day renewal
frequency excess, conditional mark scale, last mark surprise/direction, signed
run progress/persistence, expected next-mark direction and prior support.

TWSE listed-share tick sizes are .01/.05/.1/.5/1/5 across boundaries
10/50/100/500/1000 TWD (`src/sdk/trading/tick_calculator.hpp`). Native mids are
the arithmetic mean of bid and ask (`src/oms/model/quote.cpp`). Physical mids
therefore lie on a .005 TWD grid. Recover integer half-cent keys only if their
residual fits one native input ULP plus one scaling ULP; this bound must remain
below one eighth of a grid unit before rounding. Reject unsupported precision,
nonpositive/nonfinite values, ambiguous recovery or off-grid prices.
This validates the half-cent domain, not bid/ask legality from the midpoint alone.
The exact share-ladder coordinate has integer slopes 500/100/50/10/5/1 in
milliticks per half-cent. Repeated physical prices and closed paths are exact zero
even across tick bands. Original labels and their arithmetic remain untouched.

A renewal is a change in physical mid between consecutive original ten-second
origins. It is an observed grid renewal, not every intragrid quote change. New
days and missing grid intervals break the chain. The first dwell is left-censored
until a genuine observed renewal; the final observed dwell is right-censored.
Signed runs consist of consecutive renewal marks with the same exact direction;
quiet origins keep that run, reversal resets its anchor, and gaps end it. Progress
is exact signed ticks divided by five. Persistence is current run length divided
by all observed renewal marks so far that day, signed by its current direction.

All reference tables use strictly earlier completed dates, including permitted
unlabeled held-symbol history. Default history is twenty explicit calendar slots;
missing slots remain empty and never pull older replacement dates. Clock cells
are fixed thirty-minute intervals. Arrival hazard conditions additionally on
dwell age classes [0,10), [10,30), [30,60), [60,120), [120,300), [300,infinity)
seconds. Conditional mark direction uses current signed-run length classes 1,
2 and at least 3. These tables are one bounded hypothesis, not a search over
windows, parameters, horizons or symbols.

Default references require five distinct observed prior dates and thirty-two
risk intervals; conditional mark statistics require twenty marks and five prior
dates. Zero observed arrivals with supported exposure is a genuine zero
probability. Conditional size is the mean absolute millitick mark among actual
renewals, normalized by five ticks. Last mark size divides by the larger of this
prior mean at the CURRENT origin clock and one exact whole tick: it measures
the held last mark against the current expected step environment. Next-mark
direction instead conditions on the PREVIOUS renewal's clock and run length,
matching the observed prior transition table across clock boundaries.
Direction probabilities are bounded
probabilities; undefined denominators and unsupported cells remain NaN. There
is no epsilon, inverse probability, current-day reference or endpoint label.
Dwell survival uses discrete Kaplan-Meier with right censoring and becomes
missing where prior at-risk episode support is insufficient.

Configuration lives under `hypotheses.renewal`; `calendar_days` is mandatory
unless inherited from a declared `hypotheses.session_prior.calendar_days`.
The module returns the original index/order and never removes scoring origins.
Its physical-grid validation receipt is attached to the output DataFrame attrs
for inclusion in the new artifact/freeze lineage. This module does not alter V1.

V2 first tests the complete H9 mechanism as both full-baseline complement and
compact-core replacement. Its four arms are fixed in V2_PLAN.md. If this initial
mechanism qualifies, a separately frozen next comparison can distinguish
arrival/dwell-only from mark/episode-only and their combination; those internal
ablations are not added after observing V2 results. Direction gains must persist conditionally on big endpoints; magnitude
gains must occur within symbol-days and survive fixed nonoverlap/group guards.
A causal lagged-state control may test decay. A whole-day permutation that can
access future state is not evidence that availability or activity causes a gain.
If supported arrival states only reproduce ordinary volatility while marked
episodes add no direction, reject H9 rather than expanding its windows.
