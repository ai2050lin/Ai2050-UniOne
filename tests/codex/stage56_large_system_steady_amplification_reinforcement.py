from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_reinforcement_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_steady_amplification_reinforcement_summary() -> dict:
    v81 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_validation_20260321" / "summary.json"
    )
    v80 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_strengthening_20260321" / "summary.json"
    )
    brain_v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v19_20260321" / "summary.json"
    )
    bridge_v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v25_20260321" / "summary.json"
    )

    hs = v81["headline_metrics"]
    hr = v80["headline_metrics"]
    hb = brain_v19["headline_metrics"]
    ht = bridge_v25["headline_metrics"]

    steady_reinforcement_strength = _clip01(
        hs["steady_score"] * 0.34
        + hs["steady_readiness"] * 0.18
        + hr["amplification_reinforced_score"] * 0.10
        + hb["direct_brain_measure_v19"] * 0.18
        + ht["topology_training_readiness_v25"] * 0.20
    )
    steady_reinforcement_structure = _clip01(
        hs["steady_structure_stability"] * 0.40
        + hr["amplification_structure_stability"] * 0.10
        + hb["direct_structure_measure_v19"] * 0.24
        + ht["structure_rule_alignment_v25"] * 0.26
    )
    steady_reinforcement_route = _clip01(
        hs["steady_route_stability"] * 0.40
        + hr["amplification_route_stability"] * 0.10
        + hb["direct_route_measure_v19"] * 0.24
        + ht["steady_guard_v25"] * 0.26
    )
    steady_reinforcement_learning = _clip01(
        hs["steady_learning_lift"] * 0.36
        + hr["amplification_learning_lift"] * 0.08
        + hb["direct_feature_measure_v19"] * 0.16
        + ht["plasticity_rule_alignment_v25"] * 0.18
        + ht["topology_training_readiness_v25"] * 0.22
    )
    steady_reinforcement_penalty = _clip01(
        hs["steady_residual_penalty"] * 0.54
        + hb["direct_brain_gap_v19"] * 0.20
        + ht["topology_training_gap_v25"] * 0.26
    )
    steady_reinforcement_readiness = _clip01(
        (
            steady_reinforcement_strength
            + steady_reinforcement_structure
            + steady_reinforcement_route
            + steady_reinforcement_learning
            + (1.0 - steady_reinforcement_penalty)
        )
        / 5.0
    )
    steady_reinforcement_score = _clip01(
        (
            steady_reinforcement_readiness
            + steady_reinforcement_strength
            + steady_reinforcement_learning
            + (1.0 - steady_reinforcement_penalty)
        )
        / 4.0
    )
    steady_reinforcement_margin = (
        steady_reinforcement_strength
        + steady_reinforcement_structure
        + steady_reinforcement_route
        + steady_reinforcement_learning
        + steady_reinforcement_readiness
        + steady_reinforcement_score
        - steady_reinforcement_penalty
    )

    return {
        "headline_metrics": {
            "steady_reinforcement_strength": steady_reinforcement_strength,
            "steady_reinforcement_structure": steady_reinforcement_structure,
            "steady_reinforcement_route": steady_reinforcement_route,
            "steady_reinforcement_learning": steady_reinforcement_learning,
            "steady_reinforcement_penalty": steady_reinforcement_penalty,
            "steady_reinforcement_readiness": steady_reinforcement_readiness,
            "steady_reinforcement_score": steady_reinforcement_score,
            "steady_reinforcement_margin": steady_reinforcement_margin,
        },
        "steady_reinforcement_equation": {
            "strength_term": "A_steady_plus = mix(S_steady_score, R_steady, S_reinforce_score, M_brain_direct_v19, R_train_v25)",
            "structure_term": "S_steady_plus = mix(S_steady, S_reinforce, D_structure_v19, B_struct_v25)",
            "route_term": "R_steady_plus = mix(R_steady, R_reinforce, D_route_v19, H_steady_v25)",
            "learning_term": "L_steady_plus = mix(L_steady, L_reinforce, D_feature_v19, B_plastic_v25, R_train_v25)",
            "system_term": "M_steady_plus = A_steady_plus + S_steady_plus + R_steady_plus + L_steady_plus + R_steady_plus_sys - P_steady_plus",
        },
        "project_readout": {
            "summary": "更大系统稳态放大强化开始直接检验放大趋势能否从当前稳态区继续增强，而不是停在已有平台。",
            "next_question": "下一步要把这组强化结果并回脑编码直测和训练终式，确认放大趋势是否开始形成更稳定的规则层承接。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统稳态放大强化报告",
        "",
        f"- steady_reinforcement_strength: {hm['steady_reinforcement_strength']:.6f}",
        f"- steady_reinforcement_structure: {hm['steady_reinforcement_structure']:.6f}",
        f"- steady_reinforcement_route: {hm['steady_reinforcement_route']:.6f}",
        f"- steady_reinforcement_learning: {hm['steady_reinforcement_learning']:.6f}",
        f"- steady_reinforcement_penalty: {hm['steady_reinforcement_penalty']:.6f}",
        f"- steady_reinforcement_readiness: {hm['steady_reinforcement_readiness']:.6f}",
        f"- steady_reinforcement_score: {hm['steady_reinforcement_score']:.6f}",
        f"- steady_reinforcement_margin: {hm['steady_reinforcement_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_steady_amplification_reinforcement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
