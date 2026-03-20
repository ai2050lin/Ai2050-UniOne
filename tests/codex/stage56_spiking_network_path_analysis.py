from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spiking_network_path_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_spiking_network_path_summary() -> dict:
    gap = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_gap_bottleneck_analysis_20260320" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_prototype_network_readiness_20260320" / "summary.json"
    )
    feas = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320" / "summary.json"
    )
    unified = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_unified_intelligence_theory_possibility_20260320" / "summary.json"
    )

    hg = gap["headline_metrics"]
    hp = proto["headline_metrics"]
    hf = feas["headline_metrics"]
    hu = unified["headline_metrics"]

    feature_extraction_unlock = (
        hg["language_theory_completion"] + hg["brain_structure_path_readiness"]
    ) / 2.0
    structure_generation_unlock = (
        hg["brain_structure_path_readiness"] + hg["math_unification_readiness"]
    ) / 2.0
    spiking_network_path_readiness = (
        feature_extraction_unlock
        + structure_generation_unlock
        + hp["prototype_network_readiness"]
    ) / 3.0
    direct_agi_unlock = (
        hg["math_unification_readiness"]
        + hg["agi_network_realization_readiness"]
        + hu["higher_unified_intelligence_possibility"]
    ) / 3.0
    overlinearity_penalty = (
        hg["language_gap_remaining"] + hf["production_gap"] + hu["falsifiability_gap"]
    ) / 3.0

    return {
        "headline_metrics": {
            "feature_extraction_unlock": feature_extraction_unlock,
            "structure_generation_unlock": structure_generation_unlock,
            "spiking_network_path_readiness": spiking_network_path_readiness,
            "direct_agi_unlock": direct_agi_unlock,
            "overlinearity_penalty": overlinearity_penalty,
        },
        "path_equation": {
            "feature_term": "U_feature = mean(R_lang_theory, R_brain_path)",
            "structure_term": "U_structure = mean(R_brain_path, R_math_unified)",
            "path_term": "R_spike_path = mean(U_feature, U_structure, R_proto)",
            "agi_term": "U_agi = mean(R_math_unified, R_agi_net, P_unified)",
            "penalty_term": "P_linear = mean(G_lang, G_prod, D_false)",
        },
        "project_readout": {
            "summary": "语言理论补齐会显著帮助脉冲网络理解特征提取和结构生成，但不会自动把问题直接推进到 AGI 级网络落地。",
            "next_question": "下一步要验证脉冲网络路径的核心限制到底是结构生成、训练终式，还是跨模态统一仍然不足。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 脉冲网络路径分析报告",
        "",
        f"- feature_extraction_unlock: {hm['feature_extraction_unlock']:.6f}",
        f"- structure_generation_unlock: {hm['structure_generation_unlock']:.6f}",
        f"- spiking_network_path_readiness: {hm['spiking_network_path_readiness']:.6f}",
        f"- direct_agi_unlock: {hm['direct_agi_unlock']:.6f}",
        f"- overlinearity_penalty: {hm['overlinearity_penalty']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spiking_network_path_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
