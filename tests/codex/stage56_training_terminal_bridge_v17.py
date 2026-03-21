from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v17_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v17_summary() -> dict:
    v16 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v16_20260321" / "summary.json"
    )
    plateau = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_plateau_break_probe_20260321" / "summary.json"
    )
    brain_v11 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v11_20260321" / "summary.json"
    )

    hv = v16["headline_metrics"]
    hp = plateau["headline_metrics"]
    hb = brain_v11["headline_metrics"]

    plasticity_rule_alignment_v17 = _clip01(
        (
            hv["plasticity_rule_alignment_v16"]
            + hp["plateau_growth_support"]
            + (1.0 - hp["plateau_instability_penalty"])
            + hb["direct_feature_measure_v11"]
            + (1.0 - hb["direct_brain_gap_v11"])
        )
        / 5.0
    )
    structure_rule_alignment_v17 = _clip01(
        (
            hv["structure_rule_alignment_v16"]
            + hp["plateau_structure_guard"]
            + hp["plateau_route_guard"]
            + (1.0 - hp["plateau_instability_penalty"])
            + hb["direct_structure_measure_v11"]
        )
        / 5.0
    )
    topology_training_readiness_v17 = _clip01(
        (
            hv["topology_training_readiness_v16"]
            + plasticity_rule_alignment_v17
            + structure_rule_alignment_v17
            + hp["plateau_break_readiness"]
            + hb["direct_plateau_alignment_v11"]
            + (1.0 - hp["plateau_instability_penalty"])
        )
        / 6.0
    )
    topology_training_gap_v17 = max(0.0, 1.0 - topology_training_readiness_v17)
    plateau_guard_v17 = _clip01(
        (
            hp["plateau_structure_guard"]
            + hp["plateau_context_guard"]
            + hp["plateau_route_guard"]
            + topology_training_readiness_v17
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v17": plasticity_rule_alignment_v17,
            "structure_rule_alignment_v17": structure_rule_alignment_v17,
            "topology_training_readiness_v17": topology_training_readiness_v17,
            "topology_training_gap_v17": topology_training_gap_v17,
            "plateau_guard_v17": plateau_guard_v17,
        },
        "bridge_equation_v17": {
            "plasticity_term": "B_plastic_v17 = mean(B_plastic_v16, G_growth_break, 1 - P_break, D_feature_v11, 1 - G_brain_v11)",
            "structure_term": "B_struct_v17 = mean(B_struct_v16, G_struct_break, G_route_break, 1 - P_break, D_structure_v11)",
            "readiness_term": "R_train_v17 = mean(R_train_v16, B_plastic_v17, B_struct_v17, R_break, D_align_v11, 1 - P_break)",
            "gap_term": "G_train_v17 = 1 - R_train_v17",
            "guard_term": "H_break_v17 = mean(G_struct_break, G_ctx_break, G_route_break, R_train_v17)",
        },
        "project_readout": {
            "summary": "训练终式第十七桥开始直接面向平台期问题，尝试把协同护栏从补丁变成更稳定的训练规则约束。",
            "next_question": "下一步要把第十七桥并回主核，检验平台期是否真的出现松动。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十七桥报告",
        "",
        f"- plasticity_rule_alignment_v17: {hm['plasticity_rule_alignment_v17']:.6f}",
        f"- structure_rule_alignment_v17: {hm['structure_rule_alignment_v17']:.6f}",
        f"- topology_training_readiness_v17: {hm['topology_training_readiness_v17']:.6f}",
        f"- topology_training_gap_v17: {hm['topology_training_gap_v17']:.6f}",
        f"- plateau_guard_v17: {hm['plateau_guard_v17']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v17_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
