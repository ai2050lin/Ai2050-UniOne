from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v21_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v21_summary() -> dict:
    v20 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v20_20260321" / "summary.json"
    )
    persistence = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_anti_attenuation_persistence_20260321" / "summary.json"
    )
    brain_v15 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v15_20260321" / "summary.json"
    )

    hv = v20["headline_metrics"]
    hp = persistence["headline_metrics"]
    hb = brain_v15["headline_metrics"]

    plasticity_rule_alignment_v21 = _clip01(
        (
            hv["plasticity_rule_alignment_v20"]
            + hp["persistence_learning"]
            + (1.0 - hp["persistence_penalty"])
            + hb["direct_feature_measure_v15"]
            + (1.0 - hb["direct_brain_gap_v15"])
        )
        / 5.0
    )
    structure_rule_alignment_v21 = _clip01(
        (
            hv["structure_rule_alignment_v20"]
            + hp["persistence_structure"]
            + hp["persistence_route"]
            + (1.0 - hp["persistence_penalty"])
            + hb["direct_structure_measure_v15"]
        )
        / 5.0
    )
    topology_training_readiness_v21 = _clip01(
        (
            hv["topology_training_readiness_v20"]
            + plasticity_rule_alignment_v21
            + structure_rule_alignment_v21
            + hp["persistence_readiness"]
            + hb["direct_persistence_alignment_v15"]
            + (1.0 - hp["persistence_penalty"])
        )
        / 6.0
    )
    topology_training_gap_v21 = max(0.0, 1.0 - topology_training_readiness_v21)
    persistence_guard_v21 = _clip01(
        (
            hp["persistence_structure"]
            + hp["persistence_context"]
            + hp["persistence_route"]
            + topology_training_readiness_v21
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v21": plasticity_rule_alignment_v21,
            "structure_rule_alignment_v21": structure_rule_alignment_v21,
            "topology_training_readiness_v21": topology_training_readiness_v21,
            "topology_training_gap_v21": topology_training_gap_v21,
            "persistence_guard_v21": persistence_guard_v21,
        },
        "bridge_equation_v21": {
            "plasticity_term": "B_plastic_v21 = mean(B_plastic_v20, L_persist, 1 - P_persist, D_feature_v15, 1 - G_brain_v15)",
            "structure_term": "B_struct_v21 = mean(B_struct_v20, S_persist, R_persist, 1 - P_persist, D_structure_v15)",
            "readiness_term": "R_train_v21 = mean(R_train_v20, B_plastic_v21, B_struct_v21, R_persist, D_align_v15, 1 - P_persist)",
            "gap_term": "G_train_v21 = 1 - R_train_v21",
            "guard_term": "H_persist_v21 = mean(S_persist, C_persist, R_persist, R_train_v21)",
        },
        "project_readout": {
            "summary": "训练终式第二十一桥开始直接吸收反衰减持续性结果，检验训练规则层的回升能否继续稳定。",
            "next_question": "下一步要把第二十一桥并回主核，检验持续补偿是否开始固化为更稳定的规则层回升。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十一桥报告",
        "",
        f"- plasticity_rule_alignment_v21: {hm['plasticity_rule_alignment_v21']:.6f}",
        f"- structure_rule_alignment_v21: {hm['structure_rule_alignment_v21']:.6f}",
        f"- topology_training_readiness_v21: {hm['topology_training_readiness_v21']:.6f}",
        f"- topology_training_gap_v21: {hm['topology_training_gap_v21']:.6f}",
        f"- persistence_guard_v21: {hm['persistence_guard_v21']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v21_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
