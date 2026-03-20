from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_language_gap_bottleneck_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_language_gap_bottleneck_summary() -> dict:
    feas = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320" / "summary.json"
    )
    synth = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_dnn_brain_math_theory_synthesis_20260320" / "summary.json"
    )
    unified = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_unified_intelligence_theory_possibility_20260320" / "summary.json"
    )
    cross_modal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_strengthening_20260320" / "summary.json"
    )
    falsi = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_falsifiability_closure_strengthening_20260320" / "summary.json"
    )
    train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_rule_20260320" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_prototype_network_readiness_20260320" / "summary.json"
    )

    hf = feas["headline_metrics"]
    hs = synth["headline_metrics"]
    hu = unified["headline_metrics"]
    hm = cross_modal["headline_metrics"]
    hfa = falsi["headline_metrics"]
    ht = train["headline_metrics"]
    hp = proto["headline_metrics"]

    language_theory_completion = (
        hf["language_capability_readiness"]
        + _clip01(hs["dnn_language_core"] / 3.0)
        + hp["language_stack_readiness"]
    ) / 3.0
    language_gap_remaining = 1.0 - language_theory_completion

    brain_structure_path_readiness = (
        _clip01(hs["brain_encoding_core"] / 1.5)
        + hp["online_learning_readiness"]
        + hp["prototype_network_readiness"]
    ) / 3.0
    brain_path_gap = 1.0 - brain_structure_path_readiness

    math_unification_readiness = (
        hu["higher_unified_intelligence_possibility"]
        + hm["cross_modal_unification_stable"]
        + hfa["falsifiability_closure_stable"]
    ) / 3.0
    math_unification_gap = 1.0 - math_unification_readiness

    agi_network_realization_readiness = (
        hf["architecture_feasibility"]
        + ht["training_terminal_readiness"]
        + hp["prototype_network_readiness"]
    ) / 3.0
    agi_realization_gap = 1.0 - agi_network_realization_readiness

    largest_gap = max(
        language_gap_remaining,
        brain_path_gap,
        math_unification_gap,
        agi_realization_gap,
    )

    if largest_gap == language_gap_remaining:
        bottleneck_label = "language_gap"
    elif largest_gap == brain_path_gap:
        bottleneck_label = "brain_path_gap"
    elif largest_gap == math_unification_gap:
        bottleneck_label = "math_unification_gap"
    else:
        bottleneck_label = "agi_realization_gap"

    language_is_primary_bottleneck = 1.0 if bottleneck_label == "language_gap" else 0.0

    return {
        "headline_metrics": {
            "language_theory_completion": language_theory_completion,
            "language_gap_remaining": language_gap_remaining,
            "brain_structure_path_readiness": brain_structure_path_readiness,
            "brain_path_gap": brain_path_gap,
            "math_unification_readiness": math_unification_readiness,
            "math_unification_gap": math_unification_gap,
            "agi_network_realization_readiness": agi_network_realization_readiness,
            "agi_realization_gap": agi_realization_gap,
            "largest_gap": largest_gap,
            "language_is_primary_bottleneck": language_is_primary_bottleneck,
        },
        "bottleneck_equation": {
            "language_term": "R_lang_theory = mean(R_lang, T_dnn_norm, R_lang_stack)",
            "brain_term": "R_brain_path = mean(T_brain_norm, R_online_stack, R_proto)",
            "math_term": "R_math_unified = mean(P_unified, T_cross_star, C_false_star)",
            "agi_term": "R_agi_net = mean(F_arch, R_train, R_proto)",
            "bottleneck_term": "G_max = max(1-R_lang_theory, 1-R_brain_path, 1-R_math_unified, 1-R_agi_net)",
        },
        "project_readout": {
            "summary": "当前最大缺口不再主要是语言结构分析本身，而是从脑编码形成链推进到统一数学闭合和可施工网络终式的缺口。",
            "next_question": "下一步要继续验证语言理论补齐之后，是否真的会自动解锁脉冲网络结构生成；当前更像是会显著帮助，但不会单独完成整条链。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 语言缺口瓶颈分析报告",
        "",
        f"- language_theory_completion: {hm['language_theory_completion']:.6f}",
        f"- language_gap_remaining: {hm['language_gap_remaining']:.6f}",
        f"- brain_structure_path_readiness: {hm['brain_structure_path_readiness']:.6f}",
        f"- brain_path_gap: {hm['brain_path_gap']:.6f}",
        f"- math_unification_readiness: {hm['math_unification_readiness']:.6f}",
        f"- math_unification_gap: {hm['math_unification_gap']:.6f}",
        f"- agi_network_realization_readiness: {hm['agi_network_realization_readiness']:.6f}",
        f"- agi_realization_gap: {hm['agi_realization_gap']:.6f}",
        f"- largest_gap: {hm['largest_gap']:.6f}",
        f"- language_is_primary_bottleneck: {hm['language_is_primary_bottleneck']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_language_gap_bottleneck_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
