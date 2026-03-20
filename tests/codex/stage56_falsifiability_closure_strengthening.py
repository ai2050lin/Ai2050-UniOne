from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_falsifiability_closure_strengthening_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_falsifiability_closure_strengthening_summary() -> dict:
    closure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_falsifiability_closure_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )
    keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_high_retention_cross_version_validation_20260320" / "summary.json"
    )
    cross_modal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_bridge_20260320" / "summary.json"
    )

    hc = closure["headline_metrics"]
    hs = cross["headline_metrics"]
    hk = keep["headline_metrics"]
    hm = cross_modal["headline_metrics"]

    testability_strength_stable = (
        hc["testability_strength"]
        + hs["cross_version_stability_stable"]
        + hk["cross_keep_core"]
    ) / 3.0
    equation_compactness_stable = min(
        1.0,
        hc["equation_compactness"] + 0.12 * hm["cross_modal_unification_strength"],
    )
    predictive_separation_stable = (
        hc["predictive_separation"]
        + hm["action_planning_bridge"]
        + hm["language_to_general_transfer"]
    ) / 3.0
    falsifiability_closure_stable = (
        testability_strength_stable
        + equation_compactness_stable
        + predictive_separation_stable
    ) / 3.0
    residual_nonfalsifiable_stable = max(0.0, 1.0 - falsifiability_closure_stable)

    return {
        "headline_metrics": {
            "testability_strength_stable": testability_strength_stable,
            "equation_compactness_stable": equation_compactness_stable,
            "predictive_separation_stable": predictive_separation_stable,
            "falsifiability_closure_stable": falsifiability_closure_stable,
            "residual_nonfalsifiable_stable": residual_nonfalsifiable_stable,
        },
        "strengthening_equation": {
            "testability_term": "C_test_star = mean(C_test, S_cross, K_cross)",
            "compactness_term": "C_compact_star = C_compact + 0.12 * T_cross",
            "prediction_term": "C_pred_star = mean(C_pred, T_act, T_lang)",
            "closure_term": "C_false_star = mean(C_test_star, C_compact_star, C_pred_star)",
            "residual_term": "R_false_star = 1 - C_false_star",
        },
        "project_readout": {
            "summary": "可判伪闭合强化块把测试强度、方程紧凑度和预测分离度一起抬高，用来判断当前理论离可检验终式还有多远。",
            "next_question": "下一步要把这个强化后的闭合对象放进真实原型网络和跨模态任务里，确认它能在实验里保持。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 可判伪闭合强化报告",
        "",
        f"- testability_strength_stable: {hm['testability_strength_stable']:.6f}",
        f"- equation_compactness_stable: {hm['equation_compactness_stable']:.6f}",
        f"- predictive_separation_stable: {hm['predictive_separation_stable']:.6f}",
        f"- falsifiability_closure_stable: {hm['falsifiability_closure_stable']:.6f}",
        f"- residual_nonfalsifiable_stable: {hm['residual_nonfalsifiable_stable']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_falsifiability_closure_strengthening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
