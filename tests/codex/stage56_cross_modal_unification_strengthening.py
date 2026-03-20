from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_strengthening_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_cross_modal_unification_strengthening_summary() -> dict:
    bridge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_bridge_20260320" / "summary.json"
    )
    possibility = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_unified_intelligence_theory_possibility_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )
    feas = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320" / "summary.json"
    )

    hb = bridge["headline_metrics"]
    hp = possibility["headline_metrics"]
    hc = cross["headline_metrics"]
    hf = feas["headline_metrics"]

    language_to_general_stable = min(
        1.0,
        hb["language_to_general_transfer"] + 0.18 * (1.0 - hb["modality_residual"]),
    )
    modality_extension_stable = min(
        1.0,
        hb["modality_extension_strength"] + 0.12 * hp["higher_unified_intelligence_possibility"],
    )
    action_planning_stable = (
        hb["action_planning_bridge"]
        + hc["cross_version_stability_stable"]
        + hf["architecture_feasibility"]
    ) / 3.0
    cross_modal_unification_stable = (
        language_to_general_stable
        + modality_extension_stable
        + action_planning_stable
    ) / 3.0
    modality_residual_stable = max(0.0, 1.0 - cross_modal_unification_stable)

    return {
        "headline_metrics": {
            "language_to_general_stable": language_to_general_stable,
            "modality_extension_stable": modality_extension_stable,
            "action_planning_stable": action_planning_stable,
            "cross_modal_unification_stable": cross_modal_unification_stable,
            "modality_residual_stable": modality_residual_stable,
        },
        "strengthening_equation": {
            "language_term": "T_lang_star = T_lang + 0.18 * (1 - R_mod)",
            "modality_term": "T_mod_star = T_mod + 0.12 * P_unified",
            "action_term": "T_act_star = mean(T_act, S_cross, F_arch)",
            "strength_term": "T_cross_star = mean(T_lang_star, T_mod_star, T_act_star)",
            "residual_term": "R_mod_star = 1 - T_cross_star",
        },
        "project_readout": {
            "summary": "跨模态统一强化块把语言外推、模态扩展和任务桥接一起抬高，用来判断统一主线能否从语言稳定推进到更一般的智能结构。",
            "next_question": "下一步要把这个强化后的跨模态对象放进真实多任务原型里，确认它不只是理论层的平滑提升。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 跨模态统一强化报告",
        "",
        f"- language_to_general_stable: {hm['language_to_general_stable']:.6f}",
        f"- modality_extension_stable: {hm['modality_extension_stable']:.6f}",
        f"- action_planning_stable: {hm['action_planning_stable']:.6f}",
        f"- cross_modal_unification_stable: {hm['cross_modal_unification_stable']:.6f}",
        f"- modality_residual_stable: {hm['modality_residual_stable']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_cross_modal_unification_strengthening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
