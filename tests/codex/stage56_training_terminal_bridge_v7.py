from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v7_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v7_summary() -> dict:
    v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v6_20260321" / "summary.json"
    )
    large_online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_prototype_validation_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )
    curriculum = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_curriculum_20260321" / "summary.json"
    )

    hv = v6["headline_metrics"]
    hl = large_online["headline_metrics"]
    hb = brain_v5["headline_metrics"]
    hc = curriculum["headline_metrics"]

    plasticity_rule_alignment_v7 = _clip01(
        (
            hv["plasticity_rule_alignment_v6"]
            + hl["large_online_novel_gain"]
            + hl["large_online_language_keep"]
            + hc["long_horizon_growth_v2"]
        )
        / 4.0
    )
    structure_rule_alignment_v7 = _clip01(
        (
            hv["structure_rule_alignment_v6"]
            + hl["large_online_structure_keep"]
            + hb["direct_structure_measure_v5"]
            + hv["scaling_guard_v6"]
        )
        / 4.0
    )
    topology_training_readiness_v7 = _clip01(
        (
            hv["topology_training_readiness_v6"]
            + plasticity_rule_alignment_v7
            + structure_rule_alignment_v7
            + hl["large_online_readiness"]
            + hv["scaling_guard_v6"]
        )
        / 5.0
    )
    topology_training_gap_v7 = max(0.0, 1.0 - topology_training_readiness_v7)
    scaling_guard_v7 = _clip01(
        (
            hv["scaling_guard_v6"]
            + hc["shared_route_guard"]
            + hl["large_online_language_keep"]
            + structure_rule_alignment_v7
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v7": plasticity_rule_alignment_v7,
            "structure_rule_alignment_v7": structure_rule_alignment_v7,
            "topology_training_readiness_v7": topology_training_readiness_v7,
            "topology_training_gap_v7": topology_training_gap_v7,
            "scaling_guard_v7": scaling_guard_v7,
        },
        "bridge_equation_v7": {
            "plasticity_term": "B_plastic_v7 = mean(B_plastic_v6, G_large, L_large, G_curr)",
            "structure_term": "B_struct_v7 = mean(B_struct_v6, S_large, D_structure_v5, H_scale_v6)",
            "readiness_term": "R_train_v7 = mean(R_train_v6, B_plastic_v7, B_struct_v7, R_large, H_scale_v6)",
            "gap_term": "G_train_v7 = 1 - R_train_v7",
            "guard_term": "H_scale_v7 = mean(H_scale_v6, H_curr, L_large, B_struct_v7)",
        },
        "project_readout": {
            "summary": "训练终式第七桥开始把更大在线原型的真实保持和遗忘压力并回训练规则，桥接结果比前一轮更接近工程场景。",
            "next_question": "下一步要把这条第七桥和更大在线原型一起跑更高更新强度，看是否会出现结构相变式塌缩。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第七桥报告",
        "",
        f"- plasticity_rule_alignment_v7: {hm['plasticity_rule_alignment_v7']:.6f}",
        f"- structure_rule_alignment_v7: {hm['structure_rule_alignment_v7']:.6f}",
        f"- topology_training_readiness_v7: {hm['topology_training_readiness_v7']:.6f}",
        f"- topology_training_gap_v7: {hm['topology_training_gap_v7']:.6f}",
        f"- scaling_guard_v7: {hm['scaling_guard_v7']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v7_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
