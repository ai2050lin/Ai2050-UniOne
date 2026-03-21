from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v16_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v16_summary() -> dict:
    v15 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v15_20260321" / "summary.json"
    )
    coord = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coordination_stabilization_20260321" / "summary.json"
    )
    brain_v10 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v10_20260321" / "summary.json"
    )

    hv = v15["headline_metrics"]
    hc = coord["headline_metrics"]
    hb = brain_v10["headline_metrics"]

    plasticity_rule_alignment_v16 = _clip01(
        (
            hv["plasticity_rule_alignment_v15"]
            + hc["coordinated_growth_support"]
            + (1.0 - hc["coordinated_instability_penalty"])
            + hb["direct_feature_measure_v10"]
            + (1.0 - hb["direct_brain_gap_v10"])
        )
        / 5.0
    )
    structure_rule_alignment_v16 = _clip01(
        (
            hv["structure_rule_alignment_v15"]
            + hc["coordinated_structure_guard"]
            + hc["coordinated_route_guard"]
            + (1.0 - hc["coordinated_instability_penalty"])
            + hb["direct_structure_measure_v10"]
        )
        / 5.0
    )
    topology_training_readiness_v16 = _clip01(
        (
            hv["topology_training_readiness_v15"]
            + plasticity_rule_alignment_v16
            + structure_rule_alignment_v16
            + hc["coordinated_readiness"]
            + hb["direct_coord_alignment_v10"]
            + (1.0 - hc["coordinated_instability_penalty"])
        )
        / 6.0
    )
    topology_training_gap_v16 = max(0.0, 1.0 - topology_training_readiness_v16)
    coordination_guard_v16 = _clip01(
        (
            hc["coordinated_structure_guard"]
            + hc["coordinated_context_guard"]
            + hc["coordinated_route_guard"]
            + topology_training_readiness_v16
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v16": plasticity_rule_alignment_v16,
            "structure_rule_alignment_v16": structure_rule_alignment_v16,
            "topology_training_readiness_v16": topology_training_readiness_v16,
            "topology_training_gap_v16": topology_training_gap_v16,
            "coordination_guard_v16": coordination_guard_v16,
        },
        "bridge_equation_v16": {
            "plasticity_term": "B_plastic_v16 = mean(B_plastic_v15, G_growth, 1 - P_coord, D_feature_v10, 1 - G_brain_v10)",
            "structure_term": "B_struct_v16 = mean(B_struct_v15, G_struct, G_route, 1 - P_coord, D_structure_v10)",
            "readiness_term": "R_train_v16 = mean(R_train_v15, B_plastic_v16, B_struct_v16, R_coord, D_align_v10, 1 - P_coord)",
            "gap_term": "G_train_v16 = 1 - R_train_v16",
            "guard_term": "H_coord_v16 = mean(G_struct, G_ctx, G_route, R_train_v16)",
        },
        "project_readout": {
            "summary": "训练终式第十六桥开始专门吸收协同稳定化护栏，训练规则开始从单项防护转向协同防护。",
            "next_question": "下一步要把第十六桥并回主核，检验主核能否真正突破当前的平台期。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十六桥报告",
        "",
        f"- plasticity_rule_alignment_v16: {hm['plasticity_rule_alignment_v16']:.6f}",
        f"- structure_rule_alignment_v16: {hm['structure_rule_alignment_v16']:.6f}",
        f"- topology_training_readiness_v16: {hm['topology_training_readiness_v16']:.6f}",
        f"- topology_training_gap_v16: {hm['topology_training_gap_v16']:.6f}",
        f"- coordination_guard_v16: {hm['coordination_guard_v16']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v16_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
