from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v6_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v6_summary() -> dict:
    v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v5_20260321" / "summary.json"
    )
    curriculum = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_curriculum_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )

    hv = v5["headline_metrics"]
    hc = curriculum["headline_metrics"]
    hb = brain_v5["headline_metrics"]

    plasticity_rule_alignment_v6 = _clip01(
        (
            hv["plasticity_rule_alignment_v5"]
            + hc["curriculum_plasticity_gain"]
            + hc["long_horizon_growth_v2"]
            + hc["context_generalization_guard"]
        )
        / 4.0
    )
    structure_rule_alignment_v6 = _clip01(
        (
            hv["structure_rule_alignment_v5"]
            + hb["direct_structure_measure_v5"]
            + hb["direct_route_measure_v5"]
            + hc["curriculum_structural_guard"]
        )
        / 4.0
    )
    topology_training_readiness_v6 = _clip01(
        (
            hv["topology_training_readiness_v5"]
            + plasticity_rule_alignment_v6
            + structure_rule_alignment_v6
            + hb["direct_brain_measure_v5"]
            + hv["scaling_guard_v5"]
        )
        / 5.0
    )
    topology_training_gap_v6 = max(0.0, 1.0 - topology_training_readiness_v6)
    scaling_guard_v6 = _clip01(
        (
            hv["scaling_guard_v5"]
            + hc["shared_route_guard"]
            + hb["direct_topology_alignment_v5"]
            + topology_training_readiness_v6
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v6": plasticity_rule_alignment_v6,
            "structure_rule_alignment_v6": structure_rule_alignment_v6,
            "topology_training_readiness_v6": topology_training_readiness_v6,
            "topology_training_gap_v6": topology_training_gap_v6,
            "scaling_guard_v6": scaling_guard_v6,
        },
        "bridge_equation_v6": {
            "plasticity_term": "B_plastic_v6 = mean(B_plastic_v5, P_curr, G_curr, C_curr)",
            "structure_term": "B_struct_v6 = mean(B_struct_v5, D_structure_v5, D_route_v5, S_curr)",
            "readiness_term": "R_train_v6 = mean(R_train_v5, B_plastic_v6, B_struct_v6, M_brain_direct_v5, H_scale_v5)",
            "gap_term": "G_train_v6 = 1 - R_train_v6",
            "guard_term": "H_scale_v6 = mean(H_scale_v5, H_curr, A_topo_v5, R_train_v6)",
        },
        "project_readout": {
            "summary": "训练终式第六桥继续把课程式可塑性、脑编码直测第五版和规模化保护量压到同一层，训练桥开始更像真正的施工规则候选。",
            "next_question": "下一步要把这条第六桥放进更大的在线原型，直接看更高更新强度下新旧知识是否会出现系统性竞争失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第六桥报告",
        "",
        f"- plasticity_rule_alignment_v6: {hm['plasticity_rule_alignment_v6']:.6f}",
        f"- structure_rule_alignment_v6: {hm['structure_rule_alignment_v6']:.6f}",
        f"- topology_training_readiness_v6: {hm['topology_training_readiness_v6']:.6f}",
        f"- topology_training_gap_v6: {hm['topology_training_gap_v6']:.6f}",
        f"- scaling_guard_v6: {hm['scaling_guard_v6']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v6_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
