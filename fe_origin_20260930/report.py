"""Collect recorded decisions without changing selection or fitting any model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

from AstraResearch.io import ContractError, file_hash, read_yaml


def optional(path):
    return read_yaml(path) if path.exists() else None


def compact(values):
    return {
        "rows": values["rows"],
        "mean_ap": values["mean_ap"],
        "direction_auc_given_big": values["direction_auc_given_big"],
        "within_symbol_day_auc": values["mean_within_sd_auc"],
        "correct_top1": values["mean_p@0.01"],
        "opposite_top1": values["mean_wrong@0.01"],
        "net_top1": values["mean_p@0.01"] - values["mean_wrong@0.01"],
        "nonoverlap_ap": values["nonoverlap"]["mean_ap"],
    }


def number(value):
    return f"{value:.6f}" if value is not None and math.isfinite(value) else "NA"


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def decision_lines(name, decision):
    if decision is None:
        return [f"- {name}：等待結果。"]
    failed = [key for key, passed in decision["checks"].items() if not passed]
    verdict = "符合本次開發驗證門檻" if decision["predictive_gain_supported"] else "未通過本次門檻"
    lines = [f"- {name}：{verdict}；未通過項目：{', '.join(failed) or '無'}。"]
    interval = decision["daily_equal_weight_paired_AP_block_interval"]
    lines.append(f"  每日等權 AP 差，平均 {number(interval.get('mean'))}，區塊重抽樣區間 [{number(interval.get('lower'))}, {number(interval.get('upper'))}]。")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/fe_origin_20260930"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    studies = {name: root / f"study-{name}" for name in ("stock", "txf", "exf", "factorized")}
    pending = [name for name, path in studies.items() if not (path / "completed.yaml").exists()]
    if pending and not args.allow_partial:
        raise ContractError(f"Research still running: {pending}")
    summary = {"created_utc": dt.datetime.now(dt.UTC).isoformat(), "pending": pending, "pristine_OOS": False, "baseline_integrated": False, "panels": {}}
    endpoint_parity = optional(root / "native-endpoint-stock-release-1" / "results.yaml") or optional(root / "native-endpoint-stock-1" / "results.yaml")
    if endpoint_parity and set(endpoint_parity["symbols"]) == {"2609", "2337", "2481"}:
        summary["native_endpoint_parity"] = {
            "engine": endpoint_parity["engine"],
            "model_sha256": endpoint_parity["model_sha256"],
            "symbols": endpoint_parity["symbols"],
        }
    lines = [
        "# C++ FE 源頭研究：記錄結果",
        "",
        f"產生時間：{summary['created_utc']}。未完成：{', '.join(pending) or '無'}。",
        "",
        "固定目標為 60／120／180／300 秒原生 mid endpoint ±5 tick；300 秒為主要比較。所有輸入不早於 20260101，缺失日保留原角色後跳過，沒有補資料或移動角色邊界。入選門檻不含交易成本、執行或損益。",
        "",
        "股票為固定 16 個 symbol 的第一批面板，TAIFEX 為 TXF、EXF 月份合約面板。這是開發驗證；既往研究曝光仍存在，不能稱為 pristine OOS。所有 arm 的後段結果都列出，後段排名不重新決定入選者。",
        "",
    ]
    for name in ("stock", "txf", "exf"):
        path = studies[name]
        primary = optional(path / "primary-decision.yaml")
        objectives = optional(path / "objective-decisions.yaml")
        nomination = optional(path / "nomination.yaml")
        panel = {"completed": name not in pending, "nomination": nomination, "primary": primary, "objective_decisions": objectives, "scores": {}}
        lines.extend([f"## {name.upper()}", ""])
        if nomination:
            lines.append(f"校準決定的 challenger：`{nomination['challenger']}`。")
            lines.append("")
        lines.extend(decision_lines("主要比較", primary))
        if objectives:
            for arm, decision in objectives.items():
                lines.extend(decision_lines(f"{arm} 相對 {decision['regression']}", decision))
        lines.append("")
        reports = [("calibration", path / "calibration-results.yaml")]
        reports.extend((f"forward-{horizon}", path / f"forward-results-{horizon}.yaml") for horizon in (300, 60, 120, 180))
        for role, artifact in reports:
            raw = optional(artifact)
            if raw is None:
                continue
            panel["scores"][role] = {arm: compact(values) for arm, values in raw.items()}
            lines.extend(
                [
                    f"### {role}",
                    "",
                    "| Arm | AP | 大行情內方向 AUC | Symbol-day 內 AUC | 正確 Top1% | 反向 Top1% | 淨 Top1% | 不重疊 AP |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for arm, values in panel["scores"][role].items():
                row = [values[key] for key in ("mean_ap", "direction_auc_given_big", "within_symbol_day_auc", "correct_top1", "opposite_top1", "net_top1", "nonoverlap_ap")]
                lines.append(f"| {arm} | " + " | ".join(number(value) for value in row) + " |")
            lines.append("")
        panel["artifact_sha256"] = {
            p.name: file_hash(p) for p in [path / "protocol.yaml", path / "completed.yaml", path / "primary-decision.yaml", path / "objective-decisions.yaml"] if p.exists()
        }
        summary["panels"][name] = panel
    path = studies["factorized"]
    decision = optional(path / "decision.yaml")
    factorized = {
        "completed": "factorized" not in pending,
        "nomination": optional(path / "nomination.yaml"),
        "decision": decision,
        "freeze": optional(path / "freeze-receipt.yaml"),
        "scores": {},
    }
    lines.extend(
        [
            "## H5：行情幅度與条件方向分開訓練",
            "",
            "H5 使用原始 native up/down 指標，分別訓練 P(big) 與 P(up|big)，合成兩側機率。features、原始評分人口、權重來源及雙頭訓練預算與 current_binary 固定一致。這是觀察股票校準結果後提出、在股票後段評分前凍結的適應性開發假說。",
            "",
        ]
    )
    lines.extend(decision_lines("factorized 相對 current_binary", decision))
    for role in ("calibration", "forward"):
        raw = optional(path / f"{role}-results.yaml")
        if raw:
            factorized["scores"][role] = {arm: compact(values) for arm, values in raw.items()}
            lines.extend(["", f"{role}："])
            for arm, values in factorized["scores"][role].items():
                lines.append(f"- {arm}: AP {number(values['mean_ap'])}, direction AUC {number(values['direction_auc_given_big'])}, net Top1% {number(values['net_top1'])}。")
    summary["panels"]["factorized"] = factorized
    lines.extend(
        [
            "",
            "## 判讀限制與工程證據",
            "",
            "- PMQ 為現有 pmq_prefilter_v7，沒有降低門檻；stock train 102 日、3,317 numeric columns、840 selected。quality-only 為 2,807。所有 9 個新 FlowResponseSurprise 欄位因原生特殊值支持不足而被 PMQ 排除。",
            "- H2 的 flow surprise／price innovation 是嚴格先前完成區間的 joint reference；finite／unavailable 與時間置換對照保留全部原始人口及原生狀態。易推動行情的 cohort 具有幅度差異，但 train 小樣本方向未清楚分離；不能稱為已辨識因果。",
            "- H3 使用既有 CrossReturnContext；目標自身 context 與時間置換為 competing explanations。原版本的 14 欄在一個工程日依 sampling frequency 改變；修正 halt 邊界後 6 個股票同日 common-origin bitwise 檢查通過。受影響的原版 predictive material 必須重建 corrected lineage 才能整合。",
            "- 相對 representation 的缺口包含 DailyAnchor／DailyContext 在此 recipe 沒有可用 prior native daily input；metadata 宣告存在不等於模型實際看得到。",
            "- NetDepthRemovalExcessTradeRatio 有 258 個 finite 值超出 float32，因此整欄按 train domain 排除，未剪裁或刪原始列。Tortuosity 有同一實體 mid 的浮點差造成巨大比值；AsymmetricSweepImpact 的兩侧分母在不同事件 clock 衰减。來源重現寫在 SOURCE_AUDIT.md；未混入凍結輸入。",
            "- 420 registered types／1,703 families 已完成 source／parameter／export inventory。這不是每個 formula 已逐一證明正確，亦不是所有 feature 已完成獨立預測驗證。",
            "- native prefix／觀測排程／模型實際 bytes parity 與測試屬工程證據。實際 STOCK、TXF binary 頭的 C++／Python 最大機率誤差約 1.11e-16，不代表預測入選。",
            "- futures 五 tick 300 秒 event prevalence 約 89.5%／92.1%，stock 約 5.3%；futures 不等於稀有股票大行情的獨立複驗。",
            "- STOCK current_nominal 的三類模型已有明確 native CBPredictor EndpointModelPath／EndpointModelInfo route，核對實際 integer class mapping none=0／up=1／down=2，輸出原始 P(up)／P(down)。實際入選模型在 2609／2337／2481 的 3,386 個輸入與原生 labels bitwise 相同，C++／Python 最大機率誤差為 0。既有工具已支援 nominal inputs；此結果是本輪 pure endpoint recipe 相對 numeric control 的支持，不是證明所有既有 baseline 都缺少這個能力。",
            "- 完整 factory baseline 的三類 model artifact／training DAG 接線尚未整合；native route、actual-model parity 與本輪開發入選須分開判讀。H5 未通過門檻，未新增 native 合成路線。",
            "- 每日區塊重抽樣的區間是開發不確定性描述，並非多重搜尋校正、干預因果、未來交易或最終 OOS 證明。",
            "",
            "詳細動機／contract／失敗紀錄見 Experiments/fe_origin_20260930/PLAN.md、FACTORIZED_PLAN.md、SOURCE_AUDIT.md；完整原始決策與統計見各 study 子目錄。",
            "",
        ]
    )
    (root / "results-summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (root / "RESULTS.md").write_text("\n".join(lines))
    print({"report": str(root / "RESULTS.md"), "pending": pending}, flush=True)


if __name__ == "__main__":
    main()
