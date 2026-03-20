from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_primary_threshold_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_primary_threshold_closure_summary() -> dict:
    primary = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320" / "summary.json"
    )
    balance = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_balance_refinement_20260320" / "summary.json"
    )
    native = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json")

    hp = primary["headline_metrics"]
    hb = balance["headline_metrics"]
    hn = native["headline_metrics"]

    threshold_lift = hp["feature_primary_ratio"] * hn["native_selectivity"] * 4.0
    threshold_gap = 0.5 * hb["seed_normalized"] - hp["feature_structure_support"]
    primary_threshold_margin = threshold_lift - threshold_gap
    primary_threshold_ratio = (hp["feature_structure_support"] + threshold_lift) / max(0.5 * hb["seed_normalized"], 1e-9)

    return {
        "headline_metrics": {
            "threshold_lift": threshold_lift,
            "threshold_gap": threshold_gap,
            "primary_threshold_margin": primary_threshold_margin,
            "primary_threshold_ratio": primary_threshold_ratio,
        },
        "threshold_equation": {
            "lift_term": "T_lift = feature_primary_ratio * native_selectivity * 4",
            "gap_term": "T_gap = 0.5 * seed_normalized - feature_structure_support",
            "margin_term": "M_feature_threshold = T_lift - T_gap",
            "ratio_term": "R_feature_threshold = (feature_structure_support + T_lift) / (0.5 * seed_normalized)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取主结构阈值收口报告",
        "",
        f"- threshold_lift: {hm['threshold_lift']:.6f}",
        f"- threshold_gap: {hm['threshold_gap']:.6f}",
        f"- primary_threshold_margin: {hm['primary_threshold_margin']:.6f}",
        f"- primary_threshold_ratio: {hm['primary_threshold_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_primary_threshold_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
