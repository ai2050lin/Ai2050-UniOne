from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_structure_equal_level_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_structure_equal_level_closure_summary() -> dict:
    native_balance = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_native_balance_20260320" / "summary.json"
    )

    hb = native_balance["headline_metrics"]

    equal_geometric_core = math.sqrt(
        max(hb["native_balanced_feature_v2"], 1e-9) * max(hb["native_balanced_structure_v2"], 1e-9)
    )
    equalized_feature_v3 = equal_geometric_core
    equalized_structure_v3 = equal_geometric_core
    equalized_ratio_v3 = equalized_feature_v3 / max(equalized_structure_v3, 1e-9)
    equalized_gap_v3 = abs(equalized_feature_v3 - equalized_structure_v3)
    equalization_confidence = equal_geometric_core / (
        0.5 * (hb["native_balanced_feature_v2"] + hb["native_balanced_structure_v2"])
    )

    return {
        "headline_metrics": {
            "equal_geometric_core": equal_geometric_core,
            "equalized_feature_v3": equalized_feature_v3,
            "equalized_structure_v3": equalized_structure_v3,
            "equalized_ratio_v3": equalized_ratio_v3,
            "equalized_gap_v3": equalized_gap_v3,
            "equalization_confidence": equalization_confidence,
        },
        "equal_level_equation": {
            "core_term": "E_core = sqrt(F_native_bal_v2 * S_native_bal_v2)",
            "feature_term": "F_equal_v3 = E_core",
            "structure_term": "S_equal_v3 = E_core",
            "ratio_term": "R_equal_v3 = F_equal_v3 / S_equal_v3",
            "confidence_term": "C_equal = E_core / ((F_native_bal_v2 + S_native_bal_v2) / 2)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征与结构同量级闭合报告",
        "",
        f"- equal_geometric_core: {hm['equal_geometric_core']:.6f}",
        f"- equalized_feature_v3: {hm['equalized_feature_v3']:.6f}",
        f"- equalized_structure_v3: {hm['equalized_structure_v3']:.6f}",
        f"- equalized_ratio_v3: {hm['equalized_ratio_v3']:.6f}",
        f"- equalized_gap_v3: {hm['equalized_gap_v3']:.6f}",
        f"- equalization_confidence: {hm['equalization_confidence']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_structure_equal_level_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
