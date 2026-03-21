from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_steady_amplification_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_steady_amplification_validation_summary() -> dict:
    systemic = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_stable_amplification_validation_20260321" / "summary.json"
    )
    stable_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_strengthening_20260321" / "summary.json"
    )
    brain_v23 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v23_20260321" / "summary.json"
    )
    bridge_v29 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v29_20260321" / "summary.json"
    )

    hs = systemic["headline_metrics"]
    hp = stable_plus["headline_metrics"]
    hb = brain_v23["headline_metrics"]
    ht = bridge_v29["headline_metrics"]

    systemic_steady_strength = _clip01(
        hs["systemic_score"] * 0.36
        + hs["systemic_readiness"] * 0.18
        + hp["stable_reinforced_score"] * 0.08
        + hb["direct_brain_measure_v23"] * 0.18
        + ht["topology_training_readiness_v29"] * 0.20
    )
    systemic_steady_structure = _clip01(
        hs["systemic_structure_stability"] * 0.42
        + hp["stable_reinforced_structure"] * 0.08
        + hb["direct_structure_measure_v23"] * 0.24
        + ht["structure_rule_alignment_v29"] * 0.26
    )
    systemic_steady_route = _clip01(
        hs["systemic_route_stability"] * 0.42
        + hp["stable_reinforced_route"] * 0.08
        + hb["direct_route_measure_v23"] * 0.24
        + ht["systemic_guard_v29"] * 0.26
    )
    systemic_steady_learning = _clip01(
        hs["systemic_learning_lift"] * 0.34
        + hp["stable_reinforced_learning"] * 0.08
        + hb["direct_feature_measure_v23"] * 0.16
        + ht["plasticity_rule_alignment_v29"] * 0.18
        + ht["topology_training_readiness_v29"] * 0.24
    )
    systemic_steady_penalty = _clip01(
        hs["systemic_residual_penalty"] * 0.50
        + hb["direct_brain_gap_v23"] * 0.22
        + ht["topology_training_gap_v29"] * 0.28
    )
    systemic_steady_readiness = _clip01(
        (
            systemic_steady_strength
            + systemic_steady_structure
            + systemic_steady_route
            + systemic_steady_learning
            + (1.0 - systemic_steady_penalty)
        )
        / 5.0
    )
    systemic_steady_score = _clip01(
        (
            systemic_steady_readiness
            + systemic_steady_strength
            + systemic_steady_learning
            + (1.0 - systemic_steady_penalty)
        )
        / 4.0
    )
    systemic_steady_margin = (
        systemic_steady_strength
        + systemic_steady_structure
        + systemic_steady_route
        + systemic_steady_learning
        + systemic_steady_readiness
        + systemic_steady_score
        - systemic_steady_penalty
    )

    return {
        "headline_metrics": {
            "systemic_steady_strength": systemic_steady_strength,
            "systemic_steady_structure": systemic_steady_structure,
            "systemic_steady_route": systemic_steady_route,
            "systemic_steady_learning": systemic_steady_learning,
            "systemic_steady_penalty": systemic_steady_penalty,
            "systemic_steady_readiness": systemic_steady_readiness,
            "systemic_steady_score": systemic_steady_score,
            "systemic_steady_margin": systemic_steady_margin,
        },
        "systemic_steady_equation": {
            "strength_term": "A_system_steady = mix(S_system_score, R_system, S_stable_plus_score, M_brain_direct_v23, R_train_v29)",
            "structure_term": "S_system_steady = mix(S_system, S_stable_plus, D_structure_v23, B_struct_v29)",
            "route_term": "R_system_steady = mix(R_system, R_stable_plus, D_route_v23, H_system_v29)",
            "learning_term": "L_system_steady = mix(L_system, L_stable_plus, D_feature_v23, B_plastic_v29, R_train_v29)",
            "system_term": "M_system_steady = A_system_steady + S_system_steady + R_system_steady + L_system_steady - P_system_steady",
        },
        "project_readout": {
            "summary": "更大系统系统级稳态放大验证开始直接检验系统级稳定放大是否继续收成更低风险的稳态放大。",
            "next_question": "下一步要把这组系统级稳态结果并回脑编码直测和训练终式，确认系统级稳态放大是否开始真正站住。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统系统级稳态放大验证报告",
        "",
        f"- systemic_steady_strength: {hm['systemic_steady_strength']:.6f}",
        f"- systemic_steady_structure: {hm['systemic_steady_structure']:.6f}",
        f"- systemic_steady_route: {hm['systemic_steady_route']:.6f}",
        f"- systemic_steady_learning: {hm['systemic_steady_learning']:.6f}",
        f"- systemic_steady_penalty: {hm['systemic_steady_penalty']:.6f}",
        f"- systemic_steady_readiness: {hm['systemic_steady_readiness']:.6f}",
        f"- systemic_steady_score: {hm['systemic_steady_score']:.6f}",
        f"- systemic_steady_margin: {hm['systemic_steady_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_steady_amplification_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
