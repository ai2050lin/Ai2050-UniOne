from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_structure_native_balance_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_structure_native_balance_summary() -> dict:
    feature_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_terminal_direct_20260320" / "summary.json"
    )
    structure_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320" / "summary.json"
    )
    stage_summary = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_stage_summary_20260320" / "summary.json"
    )

    hf = feature_terminal["headline_metrics"]
    hs = structure_terminal["headline_metrics"]
    hg = stage_summary["headline_metrics"]

    bridge_gain = math.sqrt(
        (hs["terminal_closure_margin_v3"] + hs["terminal_feedback_closure"])
        / max(hf["feature_terminal_core_v5"] + hs["terminal_feedback_closure"], 1e-9)
    )
    native_balanced_feature_v2 = hf["feature_terminal_core_v5"] * bridge_gain * hg["convergence_smoothness"]
    native_balanced_structure_v2 = hs["terminal_closure_margin_v3"] / max(bridge_gain, 1e-9)
    native_balance_ratio_v2 = native_balanced_feature_v2 / max(native_balanced_structure_v2, 1e-9)
    native_balance_gap_v2 = abs(native_balanced_feature_v2 - native_balanced_structure_v2)

    return {
        "headline_metrics": {
            "bridge_gain": bridge_gain,
            "native_balanced_feature_v2": native_balanced_feature_v2,
            "native_balanced_structure_v2": native_balanced_structure_v2,
            "native_balance_ratio_v2": native_balance_ratio_v2,
            "native_balance_gap_v2": native_balance_gap_v2,
        },
        "native_balance_equation": {
            "bridge_term": "G_bridge = sqrt((Tc_margin + Tc_fb) / (F_terminal_v5 + Tc_fb))",
            "feature_term": "F_native_bal_v2 = F_terminal_v5 * G_bridge * convergence_smoothness",
            "structure_term": "S_native_bal_v2 = Tc_margin / G_bridge",
            "ratio_term": "R_native_bal_v2 = F_native_bal_v2 / S_native_bal_v2",
            "gap_term": "Gap_native_bal_v2 = |F_native_bal_v2 - S_native_bal_v2|",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征与结构原生平衡报告",
        "",
        f"- bridge_gain: {hm['bridge_gain']:.6f}",
        f"- native_balanced_feature_v2: {hm['native_balanced_feature_v2']:.6f}",
        f"- native_balanced_structure_v2: {hm['native_balanced_structure_v2']:.6f}",
        f"- native_balance_ratio_v2: {hm['native_balance_ratio_v2']:.6f}",
        f"- native_balance_gap_v2: {hm['native_balance_gap_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_structure_native_balance_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
