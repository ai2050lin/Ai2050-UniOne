from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_language_system_principles_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_language_system_principles_summary() -> dict:
    centrality = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_centrality_analysis_20260320" / "summary.json"
    )
    sufficiency = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_sufficiency_analysis_20260320" / "summary.json"
    )
    spike = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spiking_network_path_analysis_20260320" / "summary.json"
    )
    keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_stability_strengthening_20260320" / "summary.json"
    )
    cross_keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_high_retention_cross_version_validation_20260320" / "summary.json"
    )
    train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_rule_20260320" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_prototype_network_readiness_20260320" / "summary.json"
    )
    gap = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_gap_bottleneck_analysis_20260320" / "summary.json"
    )

    hc = centrality["headline_metrics"]
    hs = sufficiency["headline_metrics"]
    hp = spike["headline_metrics"]
    hk = keep["headline_metrics"]
    hkc = cross_keep["headline_metrics"]
    ht = train["headline_metrics"]
    hr = proto["headline_metrics"]
    hg = gap["headline_metrics"]

    language_entry_core = (
        hc["dnn_language_norm"] + hc["language_centrality"] + hg["language_theory_completion"]
    ) / 3.0
    language_feature_core = (
        hp["feature_extraction_unlock"] + hc["language_specialness"] + hc["language_bridge_power"]
    ) / 3.0
    language_structure_core = (
        hp["structure_generation_unlock"] + hk["transport_kernel_stability_stable"] + hkc["cross_keep_core"]
    ) / 3.0
    language_learning_core = (
        hs["language_only_sufficiency"] + ht["training_terminal_readiness"] + hr["prototype_network_readiness"]
    ) / 3.0
    language_pressure_core = (
        hc["language_residual"] + hs["missing_nonlanguage_mass"] + hp["overlinearity_penalty"]
    ) / 3.0
    language_system_margin = (
        language_entry_core
        + language_feature_core
        + language_structure_core
        + language_learning_core
        - language_pressure_core
    )

    return {
        "headline_metrics": {
            "language_entry_core": language_entry_core,
            "language_feature_core": language_feature_core,
            "language_structure_core": language_structure_core,
            "language_learning_core": language_learning_core,
            "language_pressure_core": language_pressure_core,
            "language_system_margin": language_system_margin,
        },
        "language_equation": {
            "entry_term": "E_lang = mean(T_dnn_norm, C_lang, R_lang_theory)",
            "feature_term": "F_lang = mean(U_feature, S_lang, B_lang)",
            "structure_term": "S_lang = mean(U_structure, K_keep_star, K_cross)",
            "learning_term": "L_lang = mean(S_lang_only, R_train, R_proto)",
            "pressure_term": "P_lang = mean(R_lang, M_nonlang, P_linear)",
            "system_term": "M_lang = E_lang + F_lang + S_lang + L_lang - P_lang",
        },
        "project_readout": {
            "summary": "语言系统当前最稳的原理已经可以压成五层：入口、特征、结构、学习、压力。语言不是静态规则集合，而是会形成、会闭合、会反馈的分层编码系统。",
            "next_question": "下一步要把这五层语言系统进一步推进到更原生的特征与回路变量，而不是只停在中层汇总对象。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 语言系统原理报告",
        "",
        f"- language_entry_core: {hm['language_entry_core']:.6f}",
        f"- language_feature_core: {hm['language_feature_core']:.6f}",
        f"- language_structure_core: {hm['language_structure_core']:.6f}",
        f"- language_learning_core: {hm['language_learning_core']:.6f}",
        f"- language_pressure_core: {hm['language_pressure_core']:.6f}",
        f"- language_system_margin: {hm['language_system_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_language_system_principles_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
