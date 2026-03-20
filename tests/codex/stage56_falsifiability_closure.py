from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_falsifiability_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_falsifiability_closure_summary() -> dict:
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )
    keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_high_retention_cross_version_validation_20260320" / "summary.json"
    )
    possibility = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_unified_intelligence_theory_possibility_20260320" / "summary.json"
    )
    feas = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320" / "summary.json"
    )

    hc = cross["headline_metrics"]
    hk = keep["headline_metrics"]
    hp = possibility["headline_metrics"]
    hf = feas["headline_metrics"]

    testability_strength = (
        hc["cross_version_stability_stable"]
        + hk["cross_keep_core"]
        + (1.0 - hf["rollback_penalty"])
    ) / 3.0
    equation_compactness = 1.0 / (1.0 + hf["production_gap"] + hf["rollback_penalty"])
    predictive_separation = hp["higher_unified_intelligence_possibility"]
    falsifiability_closure = (
        testability_strength
        + equation_compactness
        + predictive_separation
    ) / 3.0
    residual_nonfalsifiable = max(0.0, 1.0 - falsifiability_closure)

    return {
        "headline_metrics": {
            "testability_strength": testability_strength,
            "equation_compactness": equation_compactness,
            "predictive_separation": predictive_separation,
            "falsifiability_closure": falsifiability_closure,
            "residual_nonfalsifiable": residual_nonfalsifiable,
        },
        "closure_equation": {
            "testability_term": "C_test = mean(S_cross, K_cross, 1 - R_risk)",
            "compactness_term": "C_compact = 1 / (1 + G_prod + R_risk)",
            "prediction_term": "C_pred = P_unified",
            "closure_term": "C_false = mean(C_test, C_compact, C_pred)",
            "residual_term": "R_false = 1 - C_false",
        },
        "project_readout": {
            "summary": "可判伪闭合块把跨版本稳定、留核稳定和统一可能性压成了同一个闭合对象，用来判断当前理论离真正可判伪还有多远。",
            "next_question": "下一步要让这个闭合对象进入真实原型网络和跨模态任务，而不是只停在理论层的可计算值。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 可判伪闭合报告",
        "",
        f"- testability_strength: {hm['testability_strength']:.6f}",
        f"- equation_compactness: {hm['equation_compactness']:.6f}",
        f"- predictive_separation: {hm['predictive_separation']:.6f}",
        f"- falsifiability_closure: {hm['falsifiability_closure']:.6f}",
        f"- residual_nonfalsifiable: {hm['residual_nonfalsifiable']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_falsifiability_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
