from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_sustained_amplification_validation_summary() -> dict:
    v78 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v78_20260321" / "summary.json"
    )
    sustain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_rebound_validation_20260321" / "summary.json"
    )
    persistence = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_anti_attenuation_persistence_20260321" / "summary.json"
    )
    brain_v16 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v16_20260321" / "summary.json"
    )
    bridge_v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v22_20260321" / "summary.json"
    )

    hs = sustain["headline_metrics"]
    hp = persistence["headline_metrics"]
    hb = brain_v16["headline_metrics"]
    ht = bridge_v22["headline_metrics"]
    _ = v78  # keep dependency explicit for stage continuity

    amplification_structure = _clip01(
        hs["sustained_structure"] * 0.38
        + hp["persistence_structure"] * 0.14
        + hb["direct_structure_measure_v16"] * 0.24
        + ht["structure_rule_alignment_v22"] * 0.24
    )
    amplification_context = _clip01(
        hs["sustained_context"] * 0.38
        + hp["persistence_context"] * 0.14
        + hb["direct_route_measure_v16"] * 0.18
        + ht["sustained_guard_v22"] * 0.30
    )
    amplification_route = _clip01(
        hs["sustained_route"] * 0.38
        + hp["persistence_route"] * 0.14
        + hb["direct_route_measure_v16"] * 0.24
        + ht["sustained_guard_v22"] * 0.24
    )
    amplification_learning = _clip01(
        hs["sustained_learning"] * 0.35
        + hp["persistence_learning"] * 0.10
        + hb["direct_brain_measure_v16"] * 0.20
        + ht["plasticity_rule_alignment_v22"] * 0.15
        + ht["topology_training_readiness_v22"] * 0.20
    )
    amplification_penalty = _clip01(
        (
            hs["sustained_penalty"] * 0.45
            + (1.0 - ht["topology_training_readiness_v22"]) * 0.35
            + hb["direct_brain_gap_v16"] * 0.20
        )
    )
    amplification_readiness = _clip01(
        (
            amplification_structure
            + amplification_context
            + amplification_route
            + amplification_learning
            + (1.0 - amplification_penalty)
        )
        / 5.0
    )
    amplification_score = _clip01(
        (
            amplification_readiness
            + amplification_learning
            + (1.0 - amplification_penalty)
            + hs["sustained_rebound_score"]
        )
        / 4.0
    )
    amplification_margin = (
        amplification_structure
        + amplification_context
        + amplification_route
        + amplification_learning
        + amplification_readiness
        + amplification_score
        - amplification_penalty
    )

    return {
        "headline_metrics": {
            "amplification_structure": amplification_structure,
            "amplification_context": amplification_context,
            "amplification_route": amplification_route,
            "amplification_learning": amplification_learning,
            "amplification_penalty": amplification_penalty,
            "amplification_readiness": amplification_readiness,
            "amplification_score": amplification_score,
            "amplification_margin": amplification_margin,
        },
        "amplification_equation": {
            "structure_term": "S_amp = mix(S_sustain, S_persist, D_structure_v16, B_struct_v22)",
            "context_term": "C_amp = mix(C_sustain, C_persist, D_route_v16, H_sustain_v22)",
            "route_term": "R_amp = mix(R_sustain, R_persist, D_route_v16, H_sustain_v22)",
            "learning_term": "L_amp = mix(L_sustain, L_persist, M_brain_direct_v16, B_plastic_v22, R_train_v22)",
            "system_term": "M_amp = S_amp + C_amp + R_amp + L_amp + R_amp_sys - P_amp",
        },
        "project_readout": {
            "summary": "更大系统持续放大验证开始直接检测持续回升是否会继续放大，而不是停留在当前这一层。",
            "next_question": "下一步要把这组放大验证并回脑编码直测和训练终式，检查持续回升是否开始转成真正的系统级放大。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统持续放大验证报告",
        "",
        f"- amplification_structure: {hm['amplification_structure']:.6f}",
        f"- amplification_context: {hm['amplification_context']:.6f}",
        f"- amplification_route: {hm['amplification_route']:.6f}",
        f"- amplification_learning: {hm['amplification_learning']:.6f}",
        f"- amplification_penalty: {hm['amplification_penalty']:.6f}",
        f"- amplification_readiness: {hm['amplification_readiness']:.6f}",
        f"- amplification_score: {hm['amplification_score']:.6f}",
        f"- amplification_margin: {hm['amplification_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_sustained_amplification_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
