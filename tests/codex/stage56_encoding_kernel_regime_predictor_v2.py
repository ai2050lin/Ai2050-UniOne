from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_kernel_regime_predictor_v2_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _predict_group(plasticity: float, stability: float, risk: float, enc_balance: float, bind_refined: float) -> str:
    score = 0.45 * stability + 0.25 * plasticity + 0.20 * enc_balance + 0.10 * bind_refined - 0.55 * risk
    if score >= 0.72:
        return "高平衡区"
    if score <= 0.10:
        return "高风险区"
    return "过渡区"


def _actual_group(stability: float, risk: float) -> str:
    if stability >= 0.8 and risk <= 0.2:
        return "高平衡区"
    if risk >= 0.6 or stability < 0.5:
        return "高风险区"
    return "过渡区"


def build_encoding_kernel_regime_predictor_v2_summary() -> dict:
    refined = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_refinement_20260320" / "summary.json"
    )
    stability = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_stability_regime_map_20260320" / "summary.json"
    )

    hm = refined["headline_metrics"]
    rows = []
    match_count = 0
    for row in stability["rows"]:
        predicted = _predict_group(
            row["plasticity"],
            row["stability"],
            row["risk"],
            hm["encode_balance_refined"],
            hm["bind_refined"],
        )
        actual = _actual_group(row["stability"], row["risk"])
        match = predicted == actual
        if match:
            match_count += 1
        rows.append(
            {
                "name": row["name"],
                "predicted_group": predicted,
                "actual_group": actual,
                "match": match,
            }
        )

    return {
        "case_count": len(rows),
        "match_ratio": match_count / max(len(rows), 1),
        "rows": rows,
        "project_readout": {
            "summary": "这一版直接用 refined 编码核去预测高平衡区、高风险区和过渡区，检查编码机制是否比旧桥接量更接近稳态根因。",
            "next_question": "下一步要把全局统一编码核推进到资产特异回路核，减少不同资产之间的剩余误差。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 编码核到稳态区预测第二版报告",
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
    summary = build_encoding_kernel_regime_predictor_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
