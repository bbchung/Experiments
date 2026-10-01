"""Expanding nested walk-forward, with separate fit/calibration/validation roles."""

from ...io import ContractError, digest


def inner_folds(days, cfg):
    s = cfg["splits"]
    width, gap = s["validation_days"], s["purge_days"]
    folds = []
    for i in reversed(range(s["inner_folds"])):
        end = len(days) - i * width
        start = end - width
        cal_end = start - gap
        cal_start = cal_end - s["calibration_days"]
        train_end = cal_start - gap
        if train_end < s["inner_train_days"]:
            raise ContractError("Insufficient dates for precommitted inner folds; no automatic relaxation")
        folds.append({"train": days[:train_end], "calibration": days[cal_start:cal_end], "validation": days[start:end]})
    return folds


def design(days, cfg):
    s = cfg["splits"]
    folds = []
    for i in range(s["outer_folds"]):
        end = s["outer_train_days"] + i * s["outer_test_days"]
        start = end + s["purge_days"]
        stop = start + s["outer_test_days"]
        if stop > len(days):
            raise ContractError("Insufficient dates for precommitted outer folds")
        train = days[:end]
        cal_start = end - s["calibration_days"]
        fit_end = cal_start - s["purge_days"]
        fold = {
            "id": i,
            "selection_days": train,
            "inner": inner_folds(train, cfg),
            "refit": {"train": train[:fit_end], "calibration": train[cal_start:], "validation": days[start:stop]},
        }
        fold["digest"] = digest(fold)
        folds.append(fold)
    return folds
