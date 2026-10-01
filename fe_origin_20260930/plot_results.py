"""Plot the recorded magnitude/direction tradeoff; never rank new winners."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AstraResearch.io import ContractError, read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/fe_origin_20260930"))
    args = parser.parse_args()
    root = args.root.resolve()
    panels = ["stock", "txf", "exf"]
    if any(not (root / f"study-{name}/completed.yaml").exists() for name in panels):
        raise ContractError("Complete all fixed panels before plotting their recorded results")
    arms = {
        "context": ("Context", "#888888", "o"),
        "current_regression": ("Return RMSE", "#9f3b35", "s"),
        "current": ("Current 3-class", "#346a9d", "o"),
        "current_binary": ("Current binary", "#246942", "^"),
        "current_flow": ("New flow", "#bf8633", "o"),
        "current_cross": ("Cross context", "#8b4ca3", "o"),
        "current_nominal": ("Native nominal", "#b94875", "D"),
        "pmq": ("PMQ", "#26989a", "o"),
        "tail_selector": ("Tail selector", "#473e5c", "v"),
    }
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    legends = {}
    for index, name in enumerate(panels):
        study = root / f"study-{name}"
        nomination = read_yaml(study / "nomination.yaml")["challenger"]
        for column, role in enumerate(["calibration", "forward"]):
            axis = axes[index, column]
            artifact = study / ("calibration-results.yaml" if role == "calibration" else "forward-results-300.yaml")
            data = read_yaml(artifact)
            for arm, (label, color, marker) in arms.items():
                if arm not in data:
                    continue
                values = data[arm]
                point = axis.scatter(values["mean_ap"], values["direction_auc_given_big"], s=75, color=color, marker=marker, alpha=0.85, label=label)
                legends[label] = point
                if arm in {"current", "current_binary", "current_regression", nomination}:
                    axis.annotate(label, (values["mean_ap"], values["direction_auc_given_big"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
            if name == "stock" and (root / "study-factorized/completed.yaml").exists():
                result = read_yaml(root / "study-factorized" / f"{role}-results.yaml")["factorized"]
                point = axis.scatter(result["mean_ap"], result["direction_auc_given_big"], s=135, marker="*", color="#111111", label="Factorized H5")
                legends["Factorized H5"] = point
                axis.annotate("Factorized H5", (result["mean_ap"], result["direction_auc_given_big"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
            axis.axhline(0.5, color="#bbbbbb", linewidth=0.7, linestyle="--")
            axis.grid(alpha=0.2)
            axis.set_title(f"{name.upper()} | {role} | 300s endpoint, 5 ticks")
            axis.set_xlabel("Mean up/down average precision")
            axis.set_ylabel("Direction AUC among native big endpoints")
    figure.suptitle(
        "Fixed development comparisons: magnitude ranking and direction\nPoints are descriptive; admission uses predeclared calibration and paired forward guards", fontsize=12
    )
    figure.legend(list(legends.values()), list(legends), loc="outside lower center", ncol=5, fontsize=9)
    for suffix in ("png", "pdf"):
        figure.savefig(root / f"magnitude-direction.{suffix}", dpi=180)
    plt.close(figure)
    print(str(root / "magnitude-direction.png"), flush=True)


if __name__ == "__main__":
    main()
