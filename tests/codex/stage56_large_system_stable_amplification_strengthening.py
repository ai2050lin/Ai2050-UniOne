from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_strengthening_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_stable_amplification_strengthening_summary() -> dict:
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_validation_20260321" / "summary.json"
    )
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_reinforcement_20260321" / "summary.json"
    )
    brain_v21 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v21_20260321" / "summary.json"
    )
    bridge_v27 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v27_20260321" / "summary.json"
    )

    hs = stable["headline_metrics"]
    hr = reinforce["headline_metrics"]
    hb = brain_v21["headline_metrics"]
    ht = bridge_v27["headline_metrics"]

    stable_reinforced_strength = _clip01(
        hs["stable_score"] * 0.34
        + hs["stable_readiness"] * 0.18
        + hr["steady_reinforcement_score"] * 0.08
        + hb["direct_brain_measure_v21"] * 0.18
        + ht["topology_training_readiness_v27"] * 0.22
    )
    stable_reinforced_structure = _clip01(
        hs["stable_structure_stability"] * 0.40
        + hr["steady_reinforcement_structure"] * 0.08
        + hb["direct_structure_measure_v21"] * 0.24
        + ht["structure_rule_alignment_v27"] * 0.28
    )
    stable_reinforced_route = _clip01(
        hs["stable_route_stability"] * 0.40
        + hr["steady_reinforcement_route"] * 0.08
        + hb["direct_route_measure_v21"] * 0.24
        + ht["stable_guard_v27"] * 0.28
    )
    stable_reinforced_learning = _clip01(
        hs["stable_learning_lift"] * 0.34
        + hr["steady_reinforcement_learning"] * 0.08
        + hb["direct_feature_measure_v21"] * 0.16
        + ht["plasticity_rule_alignment_v27"] * 0.18
        + ht["topology_training_readiness_v27"] * 0.24
    )
    stable_reinforced_penalty = _clip01(
        hs["stable_residual_penalty"] * 0.52
        + hb["direct_brain_gap_v21"] * 0.20
        + ht["topology_training_gap_v27"] * 0.28
    )
    stable_reinforced_readiness = _clip01(
        (
            stable_reinforced_strength
            + stable_reinforced_structure
            + stable_reinforced_route
            + stable_reinforced_learning
            + (1.0 - stable_reinforced_penalty)
        )
        / 5.0
    )
    stable_reinforced_score = _clip01(
        (
            stable_reinforced_readiness
            + stable_reinforced_strength
            + stable_reinforced_learning
            + (1.0 - stable_reinforced_penalty)
        )
        / 4.0
    )
    stable_reinforced_margin = (
        stable_reinforced_strength
        + stable_reinforced_structure
        + stable_reinforced_route
        + stable_reinforced_learning
        + stable_reinforced_readiness
        + stable_reinforced_score
        - stable_reinforced_penalty
    )

    return {
        "headline_metrics": {
            "stable_reinforced_strength": stable_reinforced_strength,
            "stable_reinforced_structure": stable_reinforced_structure,
            "stable_reinforced_route": stable_reinforced_route,
            "stable_reinforced_learning": stable_reinforced_learning,
            "stable_reinforced_penalty": stable_reinforced_penalty,
            "stable_reinforced_readiness": stable_reinforced_readiness,
            "stable_reinforced_score": stable_reinforced_score,
            "stable_reinforced_margin": stable_reinforced_margin,
        },
        "stable_reinforced_equation": {
            "strength_term": "A_stable_plus = mix(S_stable_score, R_stable, S_steady_plus_score, M_brain_direct_v21, R_train_v27)",
            "structure_term": "S_stable_plus = mix(S_stable, S_steady_plus, D_structure_v21, B_struct_v27)",
            "route_term": "R_stable_plus = mix(R_stable, R_steady_plus, D_route_v21, H_stable_v27)",
            "learning_term": "L_stable_plus = mix(L_stable, L_steady_plus, D_feature_v21, B_plastic_v27, R_train_v27)",
            "system_term": "M_stable_plus = A_stable_plus + S_stable_plus + R_stable_plus + L_stable_plus + R_stable_plus_sys - P_stable_plus",
        },
        "project_readout": {
            "summary": "更大系统稳定放大强化开始直接检验稳定放大能否继续增强，而不是停在当前形成态。",
            "next_question": "下一步要把这组强化结果并回脑编码直测和训练终式，确认稳定放大是否开始形成更稳的规则层承接。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统稳定放大强化报告",
        "",
        f"- stable_reinforced_strength: {hm['stable_reinforced_strength']:.6f}",
        f"- stable_reinforced_structure: {hm['stable_reinforced_structure']:.6f}",
        f"- stable_reinforced_route: {hm['stable_reinforced_route']:.6f}",
        f"- stable_reinforced_learning: {hm['stable_reinforced_learning']:.6f}",
        f"- stable_reinforced_penalty: {hm['stable_reinforced_penalty']:.6f}",
        f"- stable_reinforced_readiness: {hm['stable_reinforced_readiness']:.6f}",
        f"- stable_reinforced_score: {hm['stable_reinforced_score']:.6f}",
        f"- stable_reinforced_margin: {hm['stable_reinforced_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_stable_amplification_strengthening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
