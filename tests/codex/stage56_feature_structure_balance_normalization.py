from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_structure_balance_normalization_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_structure_balance_normalization_summary() -> dict:
    v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v22_20260320" / "summary.json"
    )
    stage_summary = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_stage_summary_20260320" / "summary.json"
    )

    hv = v22["headline_metrics"]
    hs = stage_summary["headline_metrics"]

    feature_term = hv["feature_term_v22"]
    structure_term = hv["structure_term_v22"]

    balance_scale = math.sqrt(structure_term / max(feature_term, 1e-9))
    balanced_feature = feature_term * balance_scale
    balanced_structure = structure_term / balance_scale
    balanced_ratio = balanced_feature / max(balanced_structure, 1e-9)
    residual_gap = abs(balanced_feature - balanced_structure)
    balance_gain = hs["feature_structure_ratio"] * balance_scale

    return {
        "headline_metrics": {
            "balance_scale": balance_scale,
            "balanced_feature": balanced_feature,
            "balanced_structure": balanced_structure,
            "balanced_ratio": balanced_ratio,
            "residual_gap": residual_gap,
            "balance_gain": balance_gain,
        },
        "balance_equation": {
            "scale_term": "S_bal = sqrt(structure_term_v22 / feature_term_v22)",
            "feature_term": "F_bal = feature_term_v22 * S_bal",
            "structure_term": "S_balanced = structure_term_v22 / S_bal",
            "ratio_term": "R_bal = F_bal / S_balanced",
            "gap_term": "G_bal = |F_bal - S_balanced|",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征层与结构层量级平衡报告",
        "",
        f"- balance_scale: {hm['balance_scale']:.6f}",
        f"- balanced_feature: {hm['balanced_feature']:.6f}",
        f"- balanced_structure: {hm['balanced_structure']:.6f}",
        f"- balanced_ratio: {hm['balanced_ratio']:.6f}",
        f"- residual_gap: {hm['residual_gap']:.6f}",
        f"- balance_gain: {hm['balance_gain']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_structure_balance_normalization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
