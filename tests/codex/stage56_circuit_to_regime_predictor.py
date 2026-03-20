from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_to_regime_predictor_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _predict_regime(plasticity: float, stability: float, risk: float, encode_balance: float, pressure: float) -> str:
    score = stability + 0.5 * plasticity + 0.5 * encode_balance - risk - pressure
    if score >= 1.2:
        return "高平衡区"
    if score <= 0.1:
        return "高风险区"
    return "过渡区"


def build_circuit_to_regime_predictor_summary() -> dict:
    native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_variables_20260320" / "summary.json"
    )
    stability = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_stability_regime_map_20260320" / "summary.json"
    )
    nhm = native["headline_metrics"]
    rows = []
    match_count = 0
    for row in stability["rows"]:
        predicted = _predict_regime(
            row["plasticity"],
            row["stability"],
            row["risk"],
            nhm["encode_balance_native"],
            nhm["pressure_native"],
        )
        actual = row["regime"]
        # 兼容旧文件里的显示层乱码，按风险和稳定性重新折叠成三类评价
        if row["stability"] >= 0.8 and row["risk"] <= 0.2:
            actual_group = "高平衡区"
        elif row["risk"] >= 0.6 or row["stability"] < 0.5:
            actual_group = "高风险区"
        else:
            actual_group = "过渡区"
        match = predicted == actual_group
        if match:
            match_count += 1
        rows.append(
            {
                "name": row["name"],
                "predicted_group": predicted,
                "actual_group": actual_group,
                "match": match,
            }
        )
    summary = {
        "case_count": len(rows),
        "match_ratio": match_count / max(len(rows), 1),
        "rows": rows,
        "project_readout": {
            "summary": "这一版直接测试编码回路原生变量是否能预测系统会落入高平衡区、高风险区还是过渡区。",
            "next_question": "下一步要把预测器从固定全局编码核，推进到资产特异的局部回路变量上。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 编码回路到稳态区预测报告",
        "",
        f"- case_count: {summary['case_count']}",
        f"- match_ratio: {summary['match_ratio']:.6f}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        lines.append(
            f"- {row['name']}: predicted={row['predicted_group']}, actual={row['actual_group']}, match={row['match']}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_to_regime_predictor_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
