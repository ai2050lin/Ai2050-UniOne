from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v19_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v19_summary() -> dict:
    v18 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v18_20260321" / "summary.json"
    )
    scale_prop = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_validation_20260321" / "summary.json"
    )
    brain_v13 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v13_20260321" / "summary.json"
    )

    hv = v18["headline_metrics"]
    hs = scale_prop["headline_metrics"]
    hb = brain_v13["headline_metrics"]

    plasticity_rule_alignment_v19 = _clip01(
        (
            hv["plasticity_rule_alignment_v18"]
            + hs["scale_propagation_learning"]
            + (1.0 - hs["scale_propagation_penalty"])
            + hb["direct_feature_measure_v13"]
            + (1.0 - hb["direct_brain_gap_v13"])
        )
        / 5.0
    )
    structure_rule_alignment_v19 = _clip01(
        (
            hv["structure_rule_alignment_v18"]
            + hs["scale_propagation_structure"]
            + hs["scale_propagation_route"]
            + (1.0 - hs["scale_propagation_penalty"])
            + hb["direct_structure_measure_v13"]
        )
        / 5.0
    )
    topology_training_readiness_v19 = _clip01(
        (
            hv["topology_training_readiness_v18"]
            + plasticity_rule_alignment_v19
            + structure_rule_alignment_v19
            + hs["scale_propagation_readiness"]
            + hb["direct_scale_alignment_v13"]
            + (1.0 - hs["scale_propagation_penalty"])
        )
        / 6.0
    )
    topology_training_gap_v19 = max(0.0, 1.0 - topology_training_readiness_v19)
    scale_guard_v19 = _clip01(
        (
            hs["scale_propagation_structure"]
            + hs["scale_propagation_context"]
            + hs["scale_propagation_route"]
            + topology_training_readiness_v19
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v19": plasticity_rule_alignment_v19,
            "structure_rule_alignment_v19": structure_rule_alignment_v19,
            "topology_training_readiness_v19": topology_training_readiness_v19,
            "topology_training_gap_v19": topology_training_gap_v19,
            "scale_guard_v19": scale_guard_v19,
        },
        "bridge_equation_v19": {
            "plasticity_term": "B_plastic_v19 = mean(B_plastic_v18, L_prop_scale, 1 - P_prop_scale, D_feature_v13, 1 - G_brain_v13)",
            "structure_term": "B_struct_v19 = mean(B_struct_v18, S_prop_scale, R_prop_scale, 1 - P_prop_scale, D_structure_v13)",
            "readiness_term": "R_train_v19 = mean(R_train_v18, B_plastic_v19, B_struct_v19, R_scale, D_align_v13, 1 - P_prop_scale)",
            "gap_term": "G_train_v19 = 1 - R_train_v19",
            "guard_term": "H_scale_v19 = mean(S_prop_scale, C_prop_scale, R_prop_scale, R_train_v19)",
        },
        "project_readout": {
            "summary": "训练终式第十九桥开始直接吸收更大系统传播结果，检验平台期松动能否终于传导成训练规则层的收口。",
            "next_question": "下一步要把第十九桥并回主核，检验平台期传播是否开始形成跨层级突破。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十九桥报告",
        "",
        f"- plasticity_rule_alignment_v19: {hm['plasticity_rule_alignment_v19']:.6f}",
        f"- structure_rule_alignment_v19: {hm['structure_rule_alignment_v19']:.6f}",
        f"- topology_training_readiness_v19: {hm['topology_training_readiness_v19']:.6f}",
        f"- topology_training_gap_v19: {hm['topology_training_gap_v19']:.6f}",
        f"- scale_guard_v19: {hm['scale_guard_v19']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v19_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
