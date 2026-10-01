"""Hypothesis research under immutable, paired pure-signal contracts."""

from .contract import assert_same_comparison, assert_train_only, comparison_signature, freeze, gpu_lock, load_profile, verify

__all__ = ["assert_same_comparison", "assert_train_only", "comparison_signature", "freeze", "gpu_lock", "load_profile", "verify"]
