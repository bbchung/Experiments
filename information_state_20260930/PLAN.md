# Information-state representation research, v1

This round resumes the user's full FE goal after the prior study's pause. It is
development research for 60/120/180/300-second native mid-price endpoint moves
of at least five ticks. The goal remains a stronger representation proved under
a frozen OOS contract, not completion of this implementation or one experiment.

The official entry is `python3.13 -m astra.signal representation --profile
signal.yaml --stage <stage>`. `signal.yaml` binds the versioned `profile.yaml`.
The referenced profile owns every scientific choice; legacy fit flags cannot
override its dates, labels, horizon, sampling, loss, budget or acceptance rules.

## Information problem and hypotheses

The previous STOCK experiment had useful magnitude ranking and much weaker
conditional direction. A categorical arm improved the declared numeric control
on already-exposed forward dates, but new flow/cross values did not defeat their
availability-preserving permutation diagnostics. H5's factorized formulation
also failed its original nomination and development checks. More short flow
averages, parameter expansion, or mechanically lengthening windows do not follow
from these results.

H6 asks whether individual stocks need a target-specific view of joint price
discovery. Learn three positive peers whose 60-second return correlation stays
positive in both chronological train halves, then fit the target loading on
their mean bps movement. Separate projected common movement, target residual,
and movement of peers since the target's last sampled price change. The peer
set excludes the target. Normalize by the target's empirical train variation at
the actual 60/300-second scale, with an explicit one-current-tick resolution
floor. This differs from the existing equal mean of each peer's separately
normalized return. Return covariance is not proof of an economic link or cause.

H7 asks whether an information-arrival day differs from a stock that is ordinarily
active. Measure current absolute 60-second movement relative to the median at
the same clock on strictly earlier sessions, sustained excess activity, and
displacement since the observed origin-window anchor relative to its same-clock
prior. Use twenty original calendar slots and require five observations, skip
missing days without compressing history, and never use the current day in the
reference. Activity is a magnitude hypothesis; the signed displacement may
carry conditional direction. These are sampled-origin priors, not a claim to
observe continuous-session RV, a full-day close, or volume.

H8 asks whether persistent acceptance of a new price distribution differs from
an excursion with similar endpoint momentum. Two successive completed 120-second
blocks provide signed quantile transport, resolution-weighted coherent direction,
current price versus the recent median, proximity to the old accepted IQR, and
accepted dispersion relative to the old block. Both occupation blocks precede
the origin. This describes sampled mid residence, rather than print-volume memory
or continuously observed book occupation. Current-tick resolution scales prevent
floating midpoint cancellation from becoming full-strength direction or state.

The full candidate replaces the numerical universe with a compact semantic
control plus these nineteen state fields. Original native nominal strings remain
in all arms because prior empirical evidence justified testing their use. A
separate core-only arm tests whether apparent benefit is just pruning. A joint
within-symbol-day permutation of finite new state values retains every missing
state and core input; it is a noncausal diagnostic, never nominated or used for
OOS inference. It tests activity/availability against temporal value information.

## Frozen comparison

The native source is the previously published sixteen-stock material and its
original float64 `OriginMidPrice` metadata, with complete five-part sample-key
alignment. All history is at least 20260101. We do not reconstruct endpoints in
Python, cast labels to float32, fill missing days, or alter the native indicators.
The historical ten-second grid supplies causal feature history. All arms score
the same time-only thirty-second subset with native finite labels at all four
horizons. Feature availability never filters rows.

Train, ES, calibration and development roles keep the parent's fixed boundaries.
Four named symbols (2317, 2615, 2409, 3711) have labels excluded from fitting,
early stopping and nomination. Their unlabeled histories participate in graph
fitting and online priors: this is label-held-out, transductive development, not
an entirely unseen-symbol or pristine-OOS claim. Group metrics and guards are
reported separately. No symbols were selected using this experiment's scores.

Every arm uses the same CatBoost three-class codebook, 600-tree/depth-six budget,
seed, unweighted ES procedure, equal-symbol-day train mass and forty-calendar-day
recency half-life. No HPO or model-family comparison is performed. Seven primary
fits are serialized by one global GPU0 lock; at most six additional fits cover
the baseline and one calibration-nominated challenger (or the predeclared core
control if nomination fails) at the other three horizons.

The judge independently measures large-event AP, absolute-move ranking,
conditional-probability direction AUC, joint side AP, equal-symbol-day magnitude
ranking, fractional-boundary correct-minus-opposite Top1%, and predetermined
nonoverlapping events. Calibration requires both magnitude and joint AP to gain
at least 2%, direction AUC to gain at least 0.01 and exceed 0.52, and timing/tail/
nonoverlap guards. Development additionally requires positive three-day-block
lower bounds for daily magnitude AP, joint AP and direction, enough events, and
separate seen/label-held-out group guards. A favorable later result cannot reverse
a failed calibration nomination. Undefined daily statistics are not silently
dropped from uncertainty calculations.

The methodology is validated independently before FE comparison: native-boundary,
conditional-direction invariance, tie, cohort, exposure, role, causal-prefix and
cross-process serialization tests. The profile, complete evaluator dependencies,
native projection hashes, state arrays, fitted graph and package versions are
frozen before any fit. Every fit re-verifies the same manifest. Fitted model bytes,
feature ordering, codebook and score arrays have receipts. GPU training itself is
not assumed bitwise deterministic; frozen-model scoring is the reproducible
scientific artifact.

## Exposure and continuation

The old signal F_base/F_cross score keys prove exposure through 20260924, including
the date the older FE handoff called unused. All existing dates here are declared
development. Future final OOS begins no earlier than 20261001, after candidate
freeze, with at least thirty untouched trading days and the identical comparison
rules. The current runner deliberately has no sealed-scoring command. A future
candidate freeze and protected OOS material/scoring integration are still required.

Retain or reject each mechanism from its own diagnostics and the complete-model
comparison. If the primary hypotheses fail, use their direction/magnitude,
availability, group and horizon patterns to formulate the next distinct mechanism
round under a new declared recipe. Do not reinterpret a failure as success or
retreat to window/parameter expansion. This round does not close the active goal.

Information diffusion and persistent order flow motivate competing explanations,
not direct transfer of published effects to these horizons or Taiwan. Relevant
primary sources: [Cohen and Frazzini, Economic Links and Predictable Returns](https://pages.stern.nyu.edu/~afrazzin/pdf/Economic%20Links%20and%20Predictable%20Returns%20-%20Cohen%20and%20Frazzini.pdf),
[Bouchaud, Farmer and Lillo, How markets slowly digest changes in supply and demand](https://arxiv.org/abs/0809.0822).

