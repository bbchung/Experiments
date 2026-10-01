"""Pure signal validation: native material -> CatBoost -> big-move detection.

Trading-unaware by design: no costs, eligibility, execution or policy. The target
is the mid-price change in ticks over fixed horizons; a big move is at least
`target.threshold_ticks` ticks either way. Run with ``python3.13 -m AstraResearch.Experiments.signal``.
"""
