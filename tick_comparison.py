"""Fixed stock tick-label comparison; native labels, frozen features, no HPO.

Historical label-only comparison. Requires the frozen pre-feature-migration
binary; it must not reuse these matrices with the current tick-feature engine.
Run from AstraResearch with PYTHONPATH=. in a detached process. A failed gate
stops dependent work. Completed historical inputs are never modified.
"""

import sys as _astra_sys
from pathlib import Path as _AstraPath

_astra_repo_root = next(parent.parent for parent in _AstraPath(__file__).resolve().parents if parent.name == "AstraResearch")
_astra_sys.path.insert(0, str(_astra_repo_root))

import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import shutil
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score

from AstraResearch.units import price_ticks

KEYS = ["day", "symbol", "SampleTime", "SampleBookTime", "SampleBookSeq"]
PRICE_PROFILE = {"market": {"exchange": "TWSE"}}
FEE_TICKS = 2.0


def read(path):
    return yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(yaml.safe_dump(value, sort_keys=True))
    temp.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weights(frame, training=False):
    counts = frame.groupby(["day", "symbol"]).symbol.transform("size").to_numpy()
    if training:
        ages = (pd.Timestamp("20260810") - pd.to_datetime(frame.day)).dt.days.to_numpy()
        w = np.exp2(-ages / 40.0) / counts
        return w / w.mean()
    return 1 / (counts * frame.groupby("day").symbol.transform("nunique").to_numpy())


def spaced(frame):
    result = np.zeros(len(frame), dtype=bool)
    for _, group in frame.reset_index(drop=True).sort_values(KEYS).groupby(["day", "symbol"]):
        deadline = -np.inf
        for i, time in zip(group.index, group.SampleTime, strict=True):
            if time >= deadline:
                result[i] = True
                deadline = time + 300_000_000
    return result


def threshold(frame, column, q):
    base = frame.loc[frame.spaced]
    order = np.argsort(base[column].to_numpy(), kind="stable")
    mass = np.cumsum(weights(base)[order])
    i = min(np.searchsorted(mass / mass[-1], q), len(order) - 1)
    return float(base[column].to_numpy()[order[i]])


class Comparison:
    def __init__(self, output, previous, binary):
        self.root, self.previous, self.binary = output.resolve(), previous.resolve(), binary.resolve()
        self.design = read(previous / "design-resolved.yaml")
        self.arms = read(previous / "feature-arms.yaml")
        self.features = self.arms["rich"]
        self.v = previous.with_name("experiment-20260921-geometry-aligned-target")
        self.x = previous.with_name("experiment-20260921-barrier150-horizon30s")
        self.inventory = {}
        self.frames = {}

    def status(self, state, **details):
        value = {"state": state, "utc": datetime.now(UTC).isoformat(), "pid": os.getpid(), **details}
        write(self.root / "status.yaml", value)
        print(json.dumps(value), flush=True)

    def preflight(self, phase):
        # Only redundant feature replay parquet is pruned, immediately after
        # successful bit parity. Frozen matrices and old research are inputs.
        free = shutil.disk_usage(self.root).free
        write(self.root / "resources" / f"{phase}.yaml", {"free_bytes": free, "minimum_bytes": 40 * 2**30, "prune_policy": "verified_duplicate_feature_exports_only"})
        if free < 40 * 2**30:
            raise RuntimeError("Disk reserve below 40 GiB; preserve evidence and stop")

    def freeze(self):
        if sha(self.binary) != "7057c8bc20bb1049dbe87a2cf429437e75b26cfce7cc5b6560ad136e1f5f6bbb":
            raise RuntimeError("Historical comparison requires its frozen engine; current tick features require regenerated matrices and a new study")
        self.root.mkdir(parents=True, exist_ok=False)
        self.preflight("initial")
        paths = [self.binary, Path(__file__), Path(__file__).parents[1] / "units.py", self.previous / "jobs.yaml", self.previous / "feature-arms.yaml"]
        paths += [self.previous / "design-resolved.yaml", self.previous / "material-completion.yaml"]
        for role in self.design["roles"]:
            assert (self.v / "population" / f"{role}-features.npy").is_file()
            paths += [self.previous / "population" / f"{role}.parquet", self.previous / "fits" / f"{role}-predictions.parquet"]
            paths += [self.v / "population" / f"{role}-features.npy", self.v / "population" / f"{role}.parquet", self.x / "population" / f"{role}-extra-features.npy"]
        self.inventory = {str(p): sha(p) for p in paths if p.exists()}
        self.status("freezing_inputs")
        raw_inputs = read(self.previous / "raw-input-manifest.yaml")
        for index, (path, record) in enumerate(raw_inputs.items()):
            if index % 100 == 0:
                self.status("freezing_inputs", checked=index, total=len(raw_inputs))
            expected = record["sha256"]
            assert (sha(path) if Path(path).exists() else None) == expected, path
            if expected is not None:
                self.inventory[path] = expected
        write(self.root / "raw-input-manifest.yaml", raw_inputs)
        (self.root / "source").mkdir()
        shutil.copy2(self.binary, self.root / "source/coco")
        shutil.copy2(__file__, self.root / "source/tick_comparison.py")
        self.binary = self.root / "source/coco"
        write(self.root / "inputs.yaml", self.inventory)
        repo = Path(__file__).resolve().parents[2]
        with (self.root / "source/worktree.patch").open("w") as stream:
            subprocess.run(["git", "diff", "--", "AstraResearch", "src/oms", "src/sdk"], cwd=repo, stdout=stream, check=True)
        write(
            self.root / "design.yaml",
            {
                "barrier_ticks": 2,
                "horizon_seconds": 300,
                "round_trip_fee_tax_ticks": FEE_TICKS,
                "roles": self.design["roles"],
                "features": len(self.features),
                "fit_count": 3,
                "primary_quantile": 0.99,
                "contrast_quantile": 0.999,
                "spread_filter": None,
                "independent_oos": False,
                "native_execution_runs": 0,
                "model_parameters": self.design["model_parameters"],
                "classification_parameters": self.design["classification_parameters"],
                "fit_options": self.design["fit_options"],
                "prior_run": str(self.previous),
                "comparison": "Changed barrier task and regression target scale; same causal origins, features and model recipe. Common two-tick fee accounting.",
            },
        )

    def jobs(self):
        jobs = []
        for old in read(self.previous / "jobs.yaml"):
            if old["variant"] == "integrated_observer":
                continue
            job = {**old, "work": old["work"].replace(str(self.previous), str(self.root)), "configs": []}
            work = Path(job["work"])
            work.mkdir(parents=True)
            for source in old["configs"]:
                # Two small feature pilots are sufficient to check unchanged
                # feature bits across both train and calibration dates.
                if old["kind"] == "features" and Path(source).stem not in {"1303", "2481"}:
                    continue
                self.inventory[source] = sha(source)
                document = read(source)
                for group in document["Modules"]:
                    group["Decl"] = [
                        d for d in group["Decl"] if not d["Desc"].startswith(("QuoteBarrierLabeler.", "CBPredictor.", "ForwardReturnRateLabeler.", "TripleBarrierLabeler."))
                    ]
                    for declaration in group["Decl"]:
                        spec = declaration["Spec"]
                        if declaration["Desc"].startswith("BidAskBarrierLabeler."):
                            spec.pop("BarrierBps")
                            spec["BarrierTicks"] = 2
                        if declaration["Desc"].startswith("DatasetWriter."):
                            spec["LabelProviders"] = [p for p in spec.get("LabelProviders", []) if p.startswith("BidAskBarrierLabeler.")]
                            spec["Exports"] = [e for e in spec["Exports"] if not e.startswith("CBPredictor.")]
                path = work / Path(source).name
                write(path, document)
                job["configs"].append(str(path))
            jobs.append(job)
        write(self.root / "jobs.yaml", jobs)
        write(self.root / "inputs.yaml", self.inventory)
        return jobs

    def run_job(self, job):
        work = Path(job["work"])
        allowed = sorted(os.sched_getaffinity(0))
        cpu = allowed[job["cpu"] % len(allowed)]
        args = [
            "taskset",
            "-c",
            str(cpu),
            str(self.binary),
            "-d",
            job["day"],
            "-C",
            str(work),
            "--run-status-dir",
            str(work / "status"),
            "--log-dir",
            str(work / "logs"),
            *job["configs"],
        ]
        write(work / "command.yaml", args)
        with (work / "coco.log").open("x") as stream:
            subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=3600)
        receipts = list((work / "status").glob("*.yaml"))
        assert len(receipts) == 1 and read(receipts[0])["status"] == "completed"
        return {"day": job["day"], "variant": job["variant"], "receipt": str(receipts[0]), "receipt_sha256": sha(receipts[0])}

    def matrix(self, role, indices):
        all_features = read(self.v / "feature-arms.yaml")["matrix_features"]
        columns = [all_features.index(f) for f in self.features]
        count = pq.read_metadata(self.v / "population" / f"{role}.parquet").num_rows
        positions = np.asarray(indices, dtype="i8")
        old = positions < count
        output = np.empty((len(positions), len(columns)), dtype="f4", order="F")
        for mask, path, offset in [(old, self.v / "population" / f"{role}-features.npy", 0), (~old, self.x / "population" / f"{role}-extra-features.npy", count)]:
            if mask.any():
                source = np.load(path, mmap_mode="r")
                output[mask] = source[positions[mask] - offset][:, columns]
        return output

    def check_pilot(self, jobs):
        checks = []
        for job in jobs:
            work = Path(job["work"])
            if job["variant"] == "merged":
                continue
            for path in sorted(work.glob("data/**/*.parquet")):
                if job["kind"] == "features":
                    table = pd.read_parquet(path)
                    symbol = path.parent.name
                    old = pd.read_parquet(self.previous / "population" / f"{job['role']}.parquet")
                    old = old.loc[(old.day == job["day"]) & (old.symbol == symbol)]
                    merged = old[KEYS[2:] + ["feature_source_index"]].merge(table, on=KEYS[2:], validate="one_to_one")
                    assert len(merged) > 100
                    actual = merged[self.features].to_numpy(dtype="f4")
                    expected = self.matrix(job["role"], merged.feature_source_index)
                    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
                    checks.append({"kind": "feature_bits", "day": job["day"], "symbol": symbol, "rows": len(merged), "features": len(self.features), "output_sha256": sha(path)})
                    path.unlink()  # Verified duplicate only; keep receipt and parity proof.
                else:
                    merged = self.root / "native" / job["day"] / "merged" / path.relative_to(work)
                    pd.testing.assert_frame_equal(pd.read_parquet(path), pd.read_parquet(merged), check_exact=True)
                    checks.append({"kind": "label_separate_merged_reverse", "path": str(path), "sha256": sha(path)})
        assert any(c["kind"] == "feature_bits" for c in checks)
        assert any(c["kind"] == "label_separate_merged_reverse" for c in checks)
        write(self.root / "pilot-parity.yaml", {"status": "passed", "checks": checks})

    def native(self):
        self.preflight("native")
        jobs = self.jobs()
        pilot = [j for j in jobs if j["phase"] == "pilot"]
        results = []
        # CPU lanes avoid oversubscribing the original four replay CPUs.
        for phase, selected in [("pilot", pilot), ("main", [j for j in jobs if j["phase"] != "pilot"])]:

            def lane(cpu, batch=selected):
                return [self.run_job(j) for j in batch if j["cpu"] == cpu]

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for lane_results in pool.map(lane, sorted({j["cpu"] for j in selected})):
                    results.extend(lane_results)
                    self.status("native", phase=phase, completed=len(results), total=len(jobs))
            if phase == "pilot":
                self.check_pilot(pilot)
                self.preflight("native-main")
        write(self.root / "native-completion.yaml", {"status": "passed", "jobs": results})

    def population(self):
        self.preflight("population")
        counts = {}
        for role in self.design["roles"]:
            old = pd.read_parquet(self.previous / "population" / f"{role}.parquet")
            frames = []
            for (day, symbol), group in old.groupby(["day", "symbol"], sort=False):
                path = self.root / "native" / day / "merged/data" / day / symbol / "values.parquet"
                table = pd.read_parquet(path)
                names = [c for c in table.columns if c.startswith("bidask.")]
                frame = group.drop(columns=[c for c in group.columns if c.startswith("bidask.")]).merge(
                    table[KEYS[2:] + names], on=KEYS[2:], how="left", validate="one_to_one", indicator=True
                )
                frame["native_exported"] = frame.pop("_merge").eq("both")
                known = np.isfinite(frame["bidask.label[300s]"])
                frame["quote_observed"] = known & (frame["bidask.end_time[300s]"] <= frame.source_last_observed_event)
                frame["fit_observed"] = frame.quote_observed
                frame["gross_ticks"] = frame["bidask.long_return_ticks[300s]"]
                frame["net_ticks"] = frame.gross_ticks - FEE_TICKS
                frame["quote_hit"] = frame["bidask.label[300s]"].abs()
                frame["quote_direction"] = frame["bidask.up[300s]"]
                frame["unknown_reason"] = np.where(frame.quote_observed, "known", "native_invalid_pending_or_uncovered")
                valid = frame.loc[known]
                a, b = valid["bidask.origin_ask[300s]"], valid["bidask.origin_bid[300s]"]
                np.testing.assert_allclose(price_ticks(PRICE_PROFILE, valid["bidask.upper_barrier[300s]"]) - price_ticks(PRICE_PROFILE, a), 2, atol=1e-9)
                np.testing.assert_allclose(price_ticks(PRICE_PROFILE, b) - price_ticks(PRICE_PROFILE, valid["bidask.lower_barrier[300s]"]), 2, atol=1e-9)
                np.testing.assert_allclose(price_ticks(PRICE_PROFILE, valid["bidask.exit_bid[300s]"]) - price_ticks(PRICE_PROFILE, a), valid.gross_ticks, atol=1e-9)
                np.testing.assert_array_equal(valid["bidask.origin_book_seq[300s]"], valid.SampleBookSeq)
                assert (valid["bidask.end_time[300s]"] > valid.SampleTime).all()
                assert (valid["bidask.end_time[300s]"] <= valid.SampleTime + 300_000_000).all()
                frames.append(frame)
            frame = pd.concat(frames, ignore_index=True).set_index(KEYS).reindex(pd.MultiIndex.from_frame(old[KEYS])).reset_index()
            pd.testing.assert_frame_equal(frame[KEYS], old[KEYS])
            assert frame.feature_source_index.notna().all()
            # Do not leave legacy bps outcomes masquerading as this experiment's targets.
            frame = frame.drop(columns=[c for c in frame if "bps" in c or c in {"net_target", "native_net_target", "target_class", "native_target_class", "native_known"}])
            (self.root / "population").mkdir(exist_ok=True)
            frame.to_parquet(self.root / "population" / f"{role}.parquet", index=False)
            self.frames[role] = frame
            counts[role] = {"rows": len(frame), "known": int(frame.quote_observed.sum()), "hit_fraction_known": float(frame.loc[frame.quote_observed, "quote_hit"].mean())}
        write(self.root / "population-checks.yaml", {"same_origin_keys": True, "spread_filter": None, "roles": counts})

    def fit(self):
        results = []
        for task in ["hit", "direction", "gross"]:
            self.preflight(f"fit-{task}")
            self.status("fitting", task=task, completed=len(results), total=3)
            pools = {}
            for role in ["train", "early_stopping"]:
                frame = self.frames[role]
                mask = frame.fit_observed & ((frame.quote_hit == 1) if task == "direction" else True)
                sub = frame.loc[mask]
                label = sub["gross_ticks" if task == "gross" else f"quote_{task}"]
                pools[role] = Pool(self.matrix(role, sub.feature_source_index), label, weight=weights(sub, role == "train"), feature_names=self.features)
            model_type = CatBoostRegressor if task == "gross" else CatBoostClassifier
            parameters = self.design["model_parameters" if task == "gross" else "classification_parameters"]
            model = model_type(**parameters)
            model.fit(pools["train"], eval_set=pools["early_stopping"], **self.design["fit_options"], verbose=100)
            out = self.root / "fits" / task
            out.mkdir(parents=True)
            model.save_model(str(out / "model.cbm"))
            loaded = model_type().load_model(str(out / "model.cbm"))
            for role, frame in self.frames.items():
                pool = Pool(self.matrix(role, frame.feature_source_index), feature_names=self.features)

                def predict(m, data=pool, kind=task):
                    return m.predict(data, thread_count=4) if kind == "gross" else m.predict_proba(data, thread_count=4)[:, 1]

                values = predict(loaded)
                np.testing.assert_array_equal(values, predict(model))
                assert np.isfinite(values).all()
                frame[task + "_score"] = values
                del pool
            result = {"task": task, "trees": model.tree_count_, "best_iteration": model.get_best_iteration(), "model_sha256": sha(out / "model.cbm"), "saved_model_parity": True}
            write(out / "result.yaml", result)
            write(out / "parameters.yaml", model.get_all_params())
            write(out / "curves.yaml", model.get_evals_result())
            results.append(result)
            del pools, model, loaded
            gc.collect()
        train = self.frames["train"].loc[self.frames["train"].fit_observed]
        labels = train["bidask.label[300s]"]
        means = [float(np.average(train.loc[labels == label, "net_ticks"], weights=weights(train, True)[labels.to_numpy() == label])) for label in [1, -1, 0]]
        for role, frame in self.frames.items():
            hit, direction = frame.hit_score, frame.direction_score
            frame["coherent_score"] = hit * direction * means[0] + hit * (1 - direction) * means[1] + (1 - hit) * means[2]
            frame["direct_score"] = frame.gross_score - FEE_TICKS
            frame.to_parquet(self.root / "fits" / f"{role}-predictions.parquet", index=False)
        write(self.root / "fit-completion.yaml", {"fits": results, "train_class_net_means_ticks": means})

    def analysis(self):
        self.preflight("analysis")
        self.status("analyzing")
        # Freeze all thresholds using calibration before reading forward outcomes.
        thresholds = {}
        for kind, frame, columns in [
            ("ticks2", self.frames["calibration"], ["coherent_score", "direct_score"]),
            ("legacy60bps", pd.read_parquet(self.previous / "fits/calibration-predictions.parquet"), ["coherent_rich", "direct_rich"]),
        ]:
            for col in columns:
                for q in [0.99, 0.999]:
                    thresholds[f"{kind}/{col}/{q}"] = threshold(frame, col, q)
        write(self.root / "thresholds.yaml", thresholds)
        rows, ranking = [], []
        for role in ["calibration", "forward"]:
            frames = {"ticks2": self.frames[role], "legacy60bps": pd.read_parquet(self.previous / "fits" / f"{role}-predictions.parquet")}
            for name, cutoff in thresholds.items():
                kind, column, quantile = name.split("/")
                frame = frames[kind]
                dense = frame.loc[frame[column] >= cutoff]
                chosen = dense.loc[spaced(dense)]
                valid = chosen.loc[chosen.quote_observed]
                origin, terminal = valid["bidask.origin_ask[300s]"], valid["bidask.exit_bid[300s]"]
                gross = price_ticks(PRICE_PROFILE, terminal) - price_ticks(PRICE_PROFILE, origin)
                net = gross - FEE_TICKS
                row = {
                    "role": role,
                    "model": name,
                    "threshold": cutoff,
                    "selected": len(chosen),
                    "known": len(valid),
                    "unknown": len(chosen) - len(valid),
                    "positive_score_selected": int((chosen[column] > 0).sum()),
                }
                if len(valid):
                    row.update(
                        gross_ticks=float(np.average(gross, weights=weights(valid))),
                        net_ticks=float(np.average(net, weights=weights(valid))),
                        positive_fraction=float(np.average(net > 0, weights=weights(valid))),
                    )
                    daily = []
                    for day, group in valid.assign(realized_net_ticks=net).groupby("day"):
                        daily.append({"day": day, "net_ticks": float(np.average(group.realized_net_ticks, weights=weights(group))), "known": len(group)})
                    row["daily"] = daily
                rows.append(row)
                chosen.to_parquet(self.root / f"{role}-{kind}-{column}-q{quantile}.parquet", index=False)
            frame = frames["ticks2"]
            for task in ["hit", "direction"]:
                sub = frame.loc[frame.spaced & frame.quote_observed & ((frame.quote_hit == 1) if task == "direction" else True)]
                ranking.append({"role": role, "task": task, "rows": len(sub), "auc": float(roc_auc_score(sub[f"quote_{task}"], sub[task + "_score"], sample_weight=weights(sub)))})
        write(
            self.root / "report.yaml",
            {
                "ranking": ranking,
                "economics": rows,
                "fee_tax_ticks": FEE_TICKS,
                "scope": "development, conditional on known outcomes; hypothetical taker markout, no fill or stable OOS claim",
            },
        )
        lines = [
            "# Stock tick-label comparison",
            "",
            "Fee + tax: round-trip 2 ticks; spread already included by ask-to-bid returns. Same origins/features/recipe; 2 ticks replaces 60 bps, so this changes the target task.",
            "",
            "| Role | Model | Selected / known | Net ticks |",
            "|---|---|---:|---:|",
        ]
        for row in rows:
            lines.append(f"| {row['role']} | {row['model']} | {row['selected']} / {row['known']} | {row.get('net_ticks', float('nan')):.4f} |")
        lines += [
            "",
            "Unknown outcomes remain in selection counts. Results condition on known outcomes. Calibration thresholds stay fixed for forward. This is development evidence, not independent OOS or native execution PnL.",
        ]
        (self.root / "REPORT.md").write_text("\n".join(lines) + "\n")
        for path, expected in self.inventory.items():
            assert sha(path) == expected, f"Input changed: {path}"
        write(
            self.root / "result-manifest.yaml",
            {
                "files": {
                    str(p): sha(p)
                    for p in self.root.rglob("*")
                    if p.is_file()
                    and (p.suffix in {".yaml", ".md", ".parquet", ".cbm", ".py", ".patch"} or p == self.binary)
                    and p.name not in {"status.yaml", "result-manifest.yaml"}
                },
                "mutable_exclusions": ["status.yaml"],
            },
        )
        self.status("completed_pending_review", fits=3, next_experiment=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    study = Comparison(args.output, args.previous, args.binary)
    try:
        study.freeze()
        study.status("prepared")
        study.native()
        study.population()
        study.fit()
        study.analysis()
    except BaseException:
        if study.root.exists():
            study.status("failed", error=traceback.format_exc(), next_experiment=None)
        raise


if __name__ == "__main__":
    main()
