from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_unified_intelligence_theory_possibility_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_unified_intelligence_theory_possibility_summary() -> dict:
    bridge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_total_theory_bridge_expansion_20260320" / "summary.json"
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
    hc = cross["headline_metrics"]
    hk = keep["headline_metrics"]
    hf = feas["headline_metrics"]

    bridge_strength = hb["total_bridge_strength_expanded"]
    cross_stability = hc["cross_version_stability_stable"]
    keep_strength = hk["cross_keep_core"]
    architecture_feasibility = hf["architecture_feasibility"]
    production_gap = hf["production_gap"]
    rollback_penalty = hf["rollback_penalty"]

    unification_core = (
        bridge_strength
        + cross_stability
        + keep_strength
        + architecture_feasibility
    ) / 4.0
    first_principles_distance = 1.0 - min(
        1.0,
        (hb["brain_to_math_alignment"] + bridge_strength + cross_stability) / 3.0,
    )
    modality_gap = max(0.0, 1.0 - min(1.0, hb["brain_to_math_alignment"]))
    falsifiability_gap = (production_gap + rollback_penalty + (1.0 - cross_stability)) / 3.0
    higher_unified_intelligence_possibility = max(
        0.0,
        min(
            1.0,
            unification_core
            - 0.25 * first_principles_distance
            - 0.2 * falsifiability_gap
            - 0.1 * modality_gap,
        ),
    )

    return {
        "headline_metrics": {
            "unification_core": unification_core,
            "first_principles_distance": first_principles_distance,
            "modality_gap": modality_gap,
            "falsifiability_gap": falsifiability_gap,
            "higher_unified_intelligence_possibility": higher_unified_intelligence_possibility,
        },
        "possibility_equation": {
            "core_term": "U_core = mean(T_bridge, S_cross, K_cross, F_arch)",
            "distance_term": "D_first = 1 - mean(A_bm, T_bridge, S_cross)",
            "modality_term": "D_mod = 1 - A_bm",
            "falsifiability_term": "D_false = mean(G_prod, R_risk, 1 - S_cross)",
            "possibility_term": "P_unified = U_core - 0.25 * D_first - 0.2 * D_false - 0.1 * D_mod",
        },
        "project_readout": {
            "summary": "更高统一智能理论的可能性，当前已经不只是哲学判断，而是可以从总桥强度、跨版本稳定、高留核和工程缺口一起评估的对象。",
            "next_question": "下一步要把这个可能性对象和跨模态任务、原型网络训练结果对齐，确认它不只是语言主线里的局部乐观。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更高统一智能理论可能性报告",
        "",
        f"- unification_core: {hm['unification_core']:.6f}",
        f"- first_principles_distance: {hm['first_principles_distance']:.6f}",
        f"- modality_gap: {hm['modality_gap']:.6f}",
        f"- falsifiability_gap: {hm['falsifiability_gap']:.6f}",
        f"- higher_unified_intelligence_possibility: {hm['higher_unified_intelligence_possibility']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_unified_intelligence_theory_possibility_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
