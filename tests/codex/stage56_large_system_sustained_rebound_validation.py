from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_rebound_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_sustained_rebound_validation_summary() -> dict:
    persistence = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_anti_attenuation_persistence_20260321" / "summary.json"
    )
    scale_prop = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_validation_20260321" / "summary.json"
    )
    route_deg = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_degradation_probe_20260321" / "summary.json"
    )
    attenuation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_attenuation_probe_20260321" / "summary.json"
    )
    brain_v15 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v15_20260321" / "summary.json"
    )
    bridge_v21 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v21_20260321" / "summary.json"
    )

    hp = persistence["headline_metrics"]
    hs = scale_prop["headline_metrics"]
    hr = route_deg["headline_metrics"]
    ha = attenuation["headline_metrics"]
    hb = brain_v15["headline_metrics"]
    ht = bridge_v21["headline_metrics"]

    sustained_structure = _clip01(
        hp["persistence_structure"] * 0.35
        + hs["scale_propagation_structure"] * 0.15
        + hr["structure_resilience"] * 0.15
        + hb["direct_structure_measure_v15"] * 0.20
        + ht["structure_rule_alignment_v21"] * 0.15
    )
    sustained_context = _clip01(
        hp["persistence_context"] * 0.35
        + hs["scale_propagation_context"] * 0.15
        + (1.0 - hr["structure_phase_shift_risk"]) * 0.15
        + hb["direct_route_measure_v15"] * 0.15
        + ht["persistence_guard_v21"] * 0.20
    )
    sustained_route = _clip01(
        hp["persistence_route"] * 0.35
        + hs["scale_propagation_route"] * 0.15
        + hr["route_resilience"] * 0.15
        + hb["direct_route_measure_v15"] * 0.20
        + ht["persistence_guard_v21"] * 0.15
    )
    sustained_learning = _clip01(
        hp["persistence_learning"] * 0.30
        + hs["scale_propagation_learning"] * 0.15
        + (1.0 - ha["attenuation_learning"]) * 0.15
        + ht["topology_training_readiness_v21"] * 0.20
        + hb["direct_brain_measure_v15"] * 0.20
    )
    sustained_penalty = _clip01(
        (
            hp["persistence_penalty"]
            + hr["route_degradation_risk"]
            + hr["structure_phase_shift_risk"]
            + ht["topology_training_gap_v21"]
        )
        / 4.0
    )
    sustained_readiness = _clip01(
        (
            sustained_structure
            + sustained_context
            + sustained_route
            + sustained_learning
            + (1.0 - sustained_penalty)
        )
        / 5.0
    )
    sustained_rebound_score = _clip01(
        (
            sustained_readiness
            + sustained_learning
            + (1.0 - sustained_penalty)
            + hp["persistence_score"]
        )
        / 4.0
    )
    sustained_margin = (
        sustained_structure
        + sustained_context
        + sustained_route
        + sustained_learning
        + sustained_readiness
        + sustained_rebound_score
        - sustained_penalty
    )

    return {
        "headline_metrics": {
            "sustained_structure": sustained_structure,
            "sustained_context": sustained_context,
            "sustained_route": sustained_route,
            "sustained_learning": sustained_learning,
            "sustained_penalty": sustained_penalty,
            "sustained_readiness": sustained_readiness,
            "sustained_rebound_score": sustained_rebound_score,
            "sustained_margin": sustained_margin,
        },
        "sustained_equation": {
            "structure_term": "S_sustain = mix(S_persist, S_prop_scale, R_struct_true, D_structure_v15, B_struct_v21)",
            "context_term": "C_sustain = mix(C_persist, C_prop_scale, 1 - Q_phase_true, D_route_v15, H_persist_v21)",
            "route_term": "R_sustain = mix(R_persist, R_prop_scale, R_route_true, D_route_v15, H_persist_v21)",
            "learning_term": "L_sustain = mix(L_persist, L_prop_scale, 1 - A_learn, R_train_v21, M_brain_direct_v15)",
            "system_term": "M_sustain = S_sustain + C_sustain + R_sustain + L_sustain + R_sustain_sys - P_sustain",
        },
        "project_readout": {
            "summary": "更大系统持续回升验证开始直接检测补偿式回升能否在更大系统里继续站住，而不是重新掉回传播衰减区。",
            "next_question": "下一步要把这组持续回升结果并回脑编码直测和训练终式，检查持续化是否开始变成更稳定的系统机制。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统持续回升验证报告",
        "",
        f"- sustained_structure: {hm['sustained_structure']:.6f}",
        f"- sustained_context: {hm['sustained_context']:.6f}",
        f"- sustained_route: {hm['sustained_route']:.6f}",
        f"- sustained_learning: {hm['sustained_learning']:.6f}",
        f"- sustained_penalty: {hm['sustained_penalty']:.6f}",
        f"- sustained_readiness: {hm['sustained_readiness']:.6f}",
        f"- sustained_rebound_score: {hm['sustained_rebound_score']:.6f}",
        f"- sustained_margin: {hm['sustained_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_sustained_rebound_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
