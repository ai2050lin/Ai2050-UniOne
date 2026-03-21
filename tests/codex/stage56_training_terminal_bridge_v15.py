from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v15_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v15_summary() -> dict:
    v14 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v14_20260321" / "summary.json"
    )
    mega = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coupled_degradation_validation_20260321" / "summary.json"
    )
    brain_v9 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v9_20260321" / "summary.json"
    )

    hv = v14["headline_metrics"]
    hm = mega["headline_metrics"]
    hb = brain_v9["headline_metrics"]

    plasticity_rule_alignment_v15 = _clip01(
        (
            hv["plasticity_rule_alignment_v14"]
            + hm["mega_coupled_novel_gain"]
            + (1.0 - hm["mega_coupled_forgetting_penalty"])
            + hb["direct_feature_measure_v9"]
            + (1.0 - hb["direct_brain_gap_v9"])
        )
        / 5.0
    )
    structure_rule_alignment_v15 = _clip01(
        (
            hv["structure_rule_alignment_v14"]
            + hm["mega_coupled_structure_keep"]
            + (1.0 - hm["mega_coupled_route_degradation"])
            + (1.0 - hm["mega_coupled_collapse_risk"])
            + hb["direct_structure_measure_v9"]
        )
        / 5.0
    )
    topology_training_readiness_v15 = _clip01(
        (
            hv["topology_training_readiness_v14"]
            + plasticity_rule_alignment_v15
            + structure_rule_alignment_v15
            + hm["mega_coupled_readiness"]
            + hb["direct_mega_alignment_v9"]
            + (1.0 - hm["mega_coupled_collapse_risk"])
        )
        / 6.0
    )
    topology_training_gap_v15 = max(0.0, 1.0 - topology_training_readiness_v15)
    mega_guard_v15 = _clip01(
        (
            hm["mega_coupled_structure_keep"]
            + (1.0 - hm["mega_coupled_route_degradation"])
            + topology_training_readiness_v15
            + hb["direct_route_measure_v9"]
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v15": plasticity_rule_alignment_v15,
            "structure_rule_alignment_v15": structure_rule_alignment_v15,
            "topology_training_readiness_v15": topology_training_readiness_v15,
            "topology_training_gap_v15": topology_training_gap_v15,
            "mega_guard_v15": mega_guard_v15,
        },
        "bridge_equation_v15": {
            "plasticity_term": "B_plastic_v15 = mean(B_plastic_v14, G_mega, 1 - P_mega, D_feature_v9, 1 - G_brain_v9)",
            "structure_term": "B_struct_v15 = mean(B_struct_v14, S_mega, 1 - R_route_mega, 1 - R_collapse_mega, D_structure_v9)",
            "readiness_term": "R_train_v15 = mean(R_train_v14, B_plastic_v15, B_struct_v15, R_mega, D_align_v9, 1 - R_collapse_mega)",
            "gap_term": "G_train_v15 = 1 - R_train_v15",
            "guard_term": "H_mega_v15 = mean(S_mega, 1 - R_route_mega, R_train_v15, D_route_v9)",
        },
        "project_readout": {
            "summary": "训练终式第十五桥开始显式吸收更大系统联动退化链，使训练规则开始对规模化耦合失稳给出更直接的约束。",
            "next_question": "下一步要把第十五桥并回主核，检验主核在更大系统联动退化压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十五桥报告",
        "",
        f"- plasticity_rule_alignment_v15: {hm['plasticity_rule_alignment_v15']:.6f}",
        f"- structure_rule_alignment_v15: {hm['structure_rule_alignment_v15']:.6f}",
        f"- topology_training_readiness_v15: {hm['topology_training_readiness_v15']:.6f}",
        f"- topology_training_gap_v15: {hm['topology_training_gap_v15']:.6f}",
        f"- mega_guard_v15: {hm['mega_guard_v15']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v15_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
