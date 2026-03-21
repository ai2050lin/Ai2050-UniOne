from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_anti_attenuation_persistence_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_anti_attenuation_persistence_summary() -> dict:
    scale_prop = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_validation_20260321" / "summary.json"
    )
    attenuation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_attenuation_probe_20260321" / "summary.json"
    )
    brain_v14 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v14_20260321" / "summary.json"
    )
    bridge_v20 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v20_20260321" / "summary.json"
    )

    hs = scale_prop["headline_metrics"]
    ha = attenuation["headline_metrics"]
    hb = brain_v14["headline_metrics"]
    ht = bridge_v20["headline_metrics"]

    persistence_structure = _clip01(
        hs["scale_propagation_structure"] * 0.35
        + (1.0 - ha["attenuation_structure"]) * 0.15
        + hb["direct_structure_measure_v14"] * 0.25
        + ht["structure_rule_alignment_v20"] * 0.25
    )
    persistence_context = _clip01(
        hs["scale_propagation_context"] * 0.35
        + (1.0 - ha["attenuation_context"]) * 0.15
        + ht["anti_attenuation_guard_v20"] * 0.25
        + (1.0 - ht["topology_training_gap_v20"]) * 0.25
    )
    persistence_route = _clip01(
        hs["scale_propagation_route"] * 0.35
        + (1.0 - ha["attenuation_route"]) * 0.15
        + hb["direct_route_measure_v14"] * 0.25
        + ht["anti_attenuation_guard_v20"] * 0.25
    )
    persistence_learning = _clip01(
        hs["scale_propagation_learning"] * 0.35
        + (1.0 - ha["attenuation_learning"]) * 0.20
        + ht["plasticity_rule_alignment_v20"] * 0.25
        + (1.0 - ht["topology_training_gap_v20"]) * 0.20
    )
    persistence_penalty = _clip01(
        (
            ha["attenuation_gap"]
            + ht["topology_training_gap_v20"]
            + (1.0 - ht["anti_attenuation_guard_v20"])
        )
        / 3.0
    )
    persistence_readiness = _clip01(
        (
            persistence_structure
            + persistence_context
            + persistence_route
            + persistence_learning
            + (1.0 - persistence_penalty)
        )
        / 5.0
    )
    persistence_score = _clip01(
        (
            persistence_readiness
            + persistence_learning
            + (1.0 - persistence_penalty)
            + ha["anti_attenuation_readiness"]
        )
        / 4.0
    )
    persistence_margin = (
        persistence_structure
        + persistence_context
        + persistence_route
        + persistence_learning
        + persistence_readiness
        + persistence_score
        - persistence_penalty
    )

    return {
        "headline_metrics": {
            "persistence_structure": persistence_structure,
            "persistence_context": persistence_context,
            "persistence_route": persistence_route,
            "persistence_learning": persistence_learning,
            "persistence_penalty": persistence_penalty,
            "persistence_readiness": persistence_readiness,
            "persistence_score": persistence_score,
            "persistence_margin": persistence_margin,
        },
        "persistence_equation": {
            "structure_term": "S_persist = mix(S_prop_scale, 1 - A_struct, D_structure_v14, B_struct_v20)",
            "context_term": "C_persist = mix(C_prop_scale, 1 - A_ctx, H_anti_att_v20, 1 - G_train_v20)",
            "route_term": "R_persist = mix(R_prop_scale, 1 - A_route, D_route_v14, H_anti_att_v20)",
            "learning_term": "L_persist = mix(L_prop_scale, 1 - A_learn, B_plastic_v20, 1 - G_train_v20)",
            "system_term": "M_persist = S_persist + C_persist + R_persist + L_persist + R_persist_sys - P_persist",
        },
        "project_readout": {
            "summary": "更大系统反衰减持续性探针开始直接测补偿式回升能否继续保持，而不是只在单轮里短暂抬头。",
            "next_question": "下一步要把这组持续性结果并回脑编码直测和训练终式，检验补偿是否已经开始变成真正的持续机制。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统反衰减持续性探针报告",
        "",
        f"- persistence_structure: {hm['persistence_structure']:.6f}",
        f"- persistence_context: {hm['persistence_context']:.6f}",
        f"- persistence_route: {hm['persistence_route']:.6f}",
        f"- persistence_learning: {hm['persistence_learning']:.6f}",
        f"- persistence_penalty: {hm['persistence_penalty']:.6f}",
        f"- persistence_readiness: {hm['persistence_readiness']:.6f}",
        f"- persistence_score: {hm['persistence_score']:.6f}",
        f"- persistence_margin: {hm['persistence_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_anti_attenuation_persistence_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
