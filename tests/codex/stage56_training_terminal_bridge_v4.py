from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v4_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v4_summary() -> dict:
    bridge_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v3_20260321" / "summary.json"
    )
    plasticity = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_boost_20260321" / "summary.json"
    )
    brain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321" / "summary.json"
    )

    hb = bridge_v3["headline_metrics"]
    hp = plasticity["headline_metrics"]
    hr = brain["headline_metrics"]

    plasticity_rule_alignment_v4 = _clip01((hb["stability_rule_alignment_v3"] + hp["long_horizon_plasticity_boost"] + hp["retention_after_boost"]) / 3.0)
    structure_rule_alignment_v4 = _clip01((hb["structure_guard_strength_v3"] + hp["structural_plasticity_balance"] + hr["direct_structure_measure_v3"]) / 3.0)
    topology_training_readiness_v4 = _clip01((plasticity_rule_alignment_v4 + structure_rule_alignment_v4 + hb["topology_bridge_readiness_v3"]) / 3.0)
    topology_training_gap_v4 = max(0.0, 1.0 - topology_training_readiness_v4)

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v4": plasticity_rule_alignment_v4,
            "structure_rule_alignment_v4": structure_rule_alignment_v4,
            "topology_training_readiness_v4": topology_training_readiness_v4,
            "topology_training_gap_v4": topology_training_gap_v4,
        },
        "bridge_equation": {
            "plasticity_term": "B_plastic_v4 = mean(B_stable_v3, P_boost, R_boost)",
            "structure_term": "B_struct_v4 = mean(B_guard_v3, S_boost, D_structure_v3)",
            "readiness_term": "R_train_v4 = mean(B_plastic_v4, B_struct_v4, R_train_v3)",
            "gap_term": "G_train_v4 = 1 - R_train_v4",
        },
        "project_readout": {
            "summary": "训练终式第四桥开始把长期可塑性增强和结构保持一起并回训练规则，使训练桥不只关心稳不稳定，也关心能不能持续长新知识。",
            "next_question": "下一步要继续同时抬高结构规则对齐和可塑性规则对齐，否则系统会出现能保住但长不出，或者能长出但保不住的割裂状态。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第四桥报告",
        "",
        f"- plasticity_rule_alignment_v4: {hm['plasticity_rule_alignment_v4']:.6f}",
        f"- structure_rule_alignment_v4: {hm['structure_rule_alignment_v4']:.6f}",
        f"- topology_training_readiness_v4: {hm['topology_training_readiness_v4']:.6f}",
        f"- topology_training_gap_v4: {hm['topology_training_gap_v4']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v4_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
