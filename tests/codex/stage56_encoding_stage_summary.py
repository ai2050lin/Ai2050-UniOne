from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_stage_summary_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def build_encoding_stage_summary() -> dict:
    versions = [
        _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v17_20260320" / "summary.json"),
        _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v18_20260320" / "summary.json"),
        _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v19_20260320" / "summary.json"),
        _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v20_20260320" / "summary.json"),
        _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v21_20260320" / "summary.json"),
    ]

    margins = [v["headline_metrics"][next(k for k in v["headline_metrics"] if k.startswith("encoding_margin_"))] for v in versions]
    log_margins = [math.log(max(m, 1e-9)) for m in margins]
    log_deltas = [b - a for a, b in zip(log_margins, log_margins[1:])]

    latest = versions[-1]["headline_metrics"]
    feature_key = next(k for k in latest if k.startswith("feature_term_"))
    structure_key = next(k for k in latest if k.startswith("structure_term_"))
    learning_key = next(k for k in latest if k.startswith("learning_term_"))
    pressure_key = next(k for k in latest if k.startswith("pressure_term_"))

    convergence_smoothness = 1.0 / (1.0 + _std(log_deltas))
    feature_structure_ratio = latest[feature_key] / max(latest[structure_key], 1e-9)
    learning_pressure_ratio = latest[learning_key] / max(latest[pressure_key], 1e-9)
    stage_balance = convergence_smoothness * (1.0 + feature_structure_ratio)

    return {
        "headline_metrics": {
            "margin_v17_to_v21_mean": sum(margins) / len(margins),
            "convergence_smoothness": convergence_smoothness,
            "feature_structure_ratio": feature_structure_ratio,
            "learning_pressure_ratio": learning_pressure_ratio,
            "stage_balance": stage_balance,
        },
        "stage_summary_equation": {
            "smoothness_term": "S_conv = 1 / (1 + std(log_delta_margin))",
            "ratio_term": "R_fs = feature_term_v21 / structure_term_v21",
            "pressure_term": "R_lp = learning_term_v21 / pressure_term_v21",
            "balance_term": "B_stage = S_conv * (1 + R_fs)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制阶段摘要报告",
        "",
        f"- margin_v17_to_v21_mean: {hm['margin_v17_to_v21_mean']:.6f}",
        f"- convergence_smoothness: {hm['convergence_smoothness']:.6f}",
        f"- feature_structure_ratio: {hm['feature_structure_ratio']:.6f}",
        f"- learning_pressure_ratio: {hm['learning_pressure_ratio']:.6f}",
        f"- stage_balance: {hm['stage_balance']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_stage_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
