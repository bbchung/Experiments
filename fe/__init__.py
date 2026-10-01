"""FE research kernel v2: horizon-aware native material, frozen OOS judge.

Python orchestrates native replays, reads DatasetWriter exports, fits the fixed
CatBoost formulation and scores models. It never computes material feature
values; every model input comes from a C++ FeatureModule through the native
writer. A study profile (schema astra-fe-v2) owns the universe rule, roles,
labels, training formulation and judge; `python3.13 -m AstraResearch.Experiments.fe --help`.
"""
