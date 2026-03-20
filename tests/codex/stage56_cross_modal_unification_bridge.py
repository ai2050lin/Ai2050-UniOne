from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_bridge_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_cross_modal_unification_bridge_summary() -> dict:
    bridge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_total_theory_bridge_expansion_20260320" / "summary.json"
    )
    possibility = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_unified_intelligence_theory_possibility_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )
    keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_high_retention_cross_version_validation_20260320" / "summary.json"
    )
    feas = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320" / "summary.json"
    )

    hb = bridge["headline_metrics"]
    hp = possibility["headline_metrics"]
    hc = cross["headline_metrics"]
    hk = keep["headline_metrics"]
    hf = feas["headline_metrics"]

    language_to_general_transfer = min(
        1.0,
        0.55 * hb["total_bridge_strength_expanded"]
        + 0.25 * hp["higher_unified_intelligence_possibility"]
        + 0.20 * hb["dnn_to_brain_alignment"],
    )
    modality_extension_strength = min(
        1.0,
        0.50 * hb["brain_to_math_alignment"]
        + 0.30 * hp["unification_core"]
        + 0.20 * hc["cross_version_stability_stable"],
    )
    action_planning_bridge = (
        hc["cross_version_stability_stable"]
        + hk["cross_keep_core"]
        + hf["architecture_feasibility"]
    ) / 3.0
    cross_modal_unification_strength = (
        language_to_general_transfer
        + modality_extension_strength
        + action_planning_bridge
    ) / 3.0
    modality_residual = max(0.0, 1.0 - cross_modal_unification_strength)

    return {
        "headline_metrics": {
            "language_to_general_transfer": language_to_general_transfer,
            "modality_extension_strength": modality_extension_strength,
            "action_planning_bridge": action_planning_bridge,
            "cross_modal_unification_strength": cross_modal_unification_strength,
            "modality_residual": modality_residual,
        },
        "bridge_equation": {
            "language_term": "T_lang = 0.55 * T_bridge + 0.25 * P_unified + 0.20 * A_db",
            "modality_term": "T_mod = 0.50 * A_bm + 0.30 * U_core + 0.20 * S_cross",
            "action_term": "T_act = mean(S_cross, K_cross, F_arch)",
            "strength_term": "T_cross = mean(T_lang, T_mod, T_act)",
            "residual_term": "R_mod = 1 - T_cross",
        },
        "project_readout": {
            "summary": "跨模态统一桥把语言主线、脑编码主线和工程可行性并到了同一个对象里，用来衡量统一主线能否从语言继续外推到更一般的智能结构。",
            "next_question": "下一步要把这个跨模态桥对象放到真实多任务或多模态原型里，确认它不是只在理论层成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 跨模态统一桥报告",
        "",
        f"- language_to_general_transfer: {hm['language_to_general_transfer']:.6f}",
        f"- modality_extension_strength: {hm['modality_extension_strength']:.6f}",
        f"- action_planning_bridge: {hm['action_planning_bridge']:.6f}",
        f"- cross_modal_unification_strength: {hm['cross_modal_unification_strength']:.6f}",
        f"- modality_residual: {hm['modality_residual']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_cross_modal_unification_bridge_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
