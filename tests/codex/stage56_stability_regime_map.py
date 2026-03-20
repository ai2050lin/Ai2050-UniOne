from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_stability_regime_map_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _classify(plasticity: float, stability: float, risk: float) -> str:
    if stability >= 0.8 and risk <= 0.2:
        return "高平衡区"
    if plasticity >= 0.8 and risk >= 0.6:
        return "高风险可塑区"
    if stability < 0.5 and risk >= 0.5:
        return "脆弱漂移区"
    return "过渡区"


def build_regime_map_summary() -> dict:
    stability = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_long_horizon_stability_20260320" / "summary.json")
    rows = []
    counts = {"高平衡区": 0, "高风险可塑区": 0, "脆弱漂移区": 0, "过渡区": 0}
    for row in stability["rows"]:
        regime = _classify(row["plasticity"], row["stability"], row["risk"])
        counts[regime] += 1
        rows.append({**row, "regime": regime})
    summary = {
        "case_count": len(rows),
        "rows": rows,
        "headline_metrics": counts,
        "project_readout": {
            "summary": (
                "这一步把长期在线资产从连续数值坐标进一步压成稳定区、风险区和过渡区。"
                "目标是判断当前系统是否已经开始出现可重复的长期学习相区。"
            ),
            "next_question": "下一步要看这些相区是否和 G/S/L_base/L_select 以及训练阶段顺序有稳定对应。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 长期在线稳态分区报告",
        "",
        f"- 高平衡区: {summary['headline_metrics']['高平衡区']}",
        f"- 高风险可塑区: {summary['headline_metrics']['高风险可塑区']}",
        f"- 脆弱漂移区: {summary['headline_metrics']['脆弱漂移区']}",
        f"- 过渡区: {summary['headline_metrics']['过渡区']}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        lines.append(
            f"- {row['name']}: regime={row['regime']}, plasticity={row['plasticity']:.6f}, "
            f"stability={row['stability']:.6f}, risk={row['risk']:.6f}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_regime_map_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
