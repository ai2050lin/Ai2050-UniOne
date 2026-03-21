from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v5_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v5_summary() -> dict:
    bridge_v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v4_20260321" / "summary.json"
    )
    reinforcement = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_reinforcement_20260321" / "summary.json"
    )
    brain_v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v4_20260321" / "summary.json"
    )

    hb = bridge_v4["headline_metrics"]
    hr = reinforcement["headline_metrics"]
    hd = brain_v4["headline_metrics"]

    plasticity_rule_alignment_v5 = _clip01(
        (
            hb["plasticity_rule_alignment_v4"]
            + hr["adaptive_plasticity_gain"]
            + hr["structural_retention_reinforced"]
            + hr["contextual_retention_reinforced"]
        )
        / 4.0
    )
    structure_rule_alignment_v5 = _clip01(
        (
            hb["structure_rule_alignment_v4"]
            + hd["direct_structure_measure_v4"]
            + hd["direct_route_measure_v4"]
            + hr["structural_retention_reinforced"]
        )
        / 4.0
    )
    topology_training_readiness_v5 = _clip01(
        (
            hb["topology_training_readiness_v4"]
            + plasticity_rule_alignment_v5
            + structure_rule_alignment_v5
            + hd["direct_brain_measure_v4"]
        )
        / 4.0
    )
    topology_training_gap_v5 = max(0.0, 1.0 - topology_training_readiness_v5)
    scaling_guard_v5 = _clip01(
        (hr["shared_guard_reinforced"] + hd["dynamic_structure_balance_v4"] + topology_training_readiness_v5) / 3.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v5": plasticity_rule_alignment_v5,
            "structure_rule_alignment_v5": structure_rule_alignment_v5,
            "topology_training_readiness_v5": topology_training_readiness_v5,
            "topology_training_gap_v5": topology_training_gap_v5,
            "scaling_guard_v5": scaling_guard_v5,
        },
        "bridge_equation_v5": {
            "plasticity_term": "B_plastic_v5 = mean(B_plastic_v4, P_adapt, R_struct_plus, C_ctx_plus)",
            "structure_term": "B_struct_v5 = mean(B_struct_v4, D_structure_v4, D_route_v4, R_struct_plus)",
            "readiness_term": "R_train_v5 = mean(R_train_v4, B_plastic_v5, B_struct_v5, M_brain_direct_v4)",
            "gap_term": "G_train_v5 = 1 - R_train_v5",
            "guard_term": "H_scale_v5 = mean(H_guard_plus, S_balance_v4, R_train_v5)",
        },
        "project_readout": {
            "summary": "训练终式第五桥把长期可塑性强化、脑编码直测第四版和训练规则并到同一层，开始同时约束增量学习、结构保持和规模化防塌缩。",
            "next_question": "下一步要继续压低训练桥缺口，并把这条桥放进更大的在线原型里检验是否会出现新旧知识竞争失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第五桥报告",
        "",
        f"- plasticity_rule_alignment_v5: {hm['plasticity_rule_alignment_v5']:.6f}",
        f"- structure_rule_alignment_v5: {hm['structure_rule_alignment_v5']:.6f}",
        f"- topology_training_readiness_v5: {hm['topology_training_readiness_v5']:.6f}",
        f"- topology_training_gap_v5: {hm['topology_training_gap_v5']:.6f}",
        f"- scaling_guard_v5: {hm['scaling_guard_v5']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v5_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
