from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v18_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v18_summary() -> dict:
    v17 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v17_20260321" / "summary.json"
    )
    propagation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_plateau_break_propagation_probe_20260321" / "summary.json"
    )
    brain_v12 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v12_20260321" / "summary.json"
    )

    hv = v17["headline_metrics"]
    hp = propagation["headline_metrics"]
    hb = brain_v12["headline_metrics"]

    plasticity_rule_alignment_v18 = _clip01(
        (
            hv["plasticity_rule_alignment_v17"]
            + hp["propagation_learning"]
            + (1.0 - hp["propagation_penalty"])
            + hb["direct_feature_measure_v12"]
            + (1.0 - hb["direct_brain_gap_v12"])
        )
        / 5.0
    )
    structure_rule_alignment_v18 = _clip01(
        (
            hv["structure_rule_alignment_v17"]
            + hp["propagation_structure"]
            + hp["propagation_route"]
            + (1.0 - hp["propagation_penalty"])
            + hb["direct_structure_measure_v12"]
        )
        / 5.0
    )
    topology_training_readiness_v18 = _clip01(
        (
            hv["topology_training_readiness_v17"]
            + plasticity_rule_alignment_v18
            + structure_rule_alignment_v18
            + hp["propagation_readiness"]
            + hb["direct_propagation_alignment_v12"]
            + (1.0 - hp["propagation_penalty"])
        )
        / 6.0
    )
    topology_training_gap_v18 = max(0.0, 1.0 - topology_training_readiness_v18)
    propagation_guard_v18 = _clip01(
        (
            hp["propagation_structure"]
            + hp["propagation_context"]
            + hp["propagation_route"]
            + topology_training_readiness_v18
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v18": plasticity_rule_alignment_v18,
            "structure_rule_alignment_v18": structure_rule_alignment_v18,
            "topology_training_readiness_v18": topology_training_readiness_v18,
            "topology_training_gap_v18": topology_training_gap_v18,
            "propagation_guard_v18": propagation_guard_v18,
        },
        "bridge_equation_v18": {
            "plasticity_term": "B_plastic_v18 = mean(B_plastic_v17, T_learn, 1 - P_prop, D_feature_v12, 1 - G_brain_v12)",
            "structure_term": "B_struct_v18 = mean(B_struct_v17, T_struct, T_route, 1 - P_prop, D_structure_v12)",
            "readiness_term": "R_train_v18 = mean(R_train_v17, B_plastic_v18, B_struct_v18, R_prop, D_align_v12, 1 - P_prop)",
            "gap_term": "G_train_v18 = 1 - R_train_v18",
            "guard_term": "H_prop_v18 = mean(T_struct, T_ctx, T_route, R_train_v18)",
        },
        "project_readout": {
            "summary": "训练终式第十八桥开始直接吸收传播级破平台结果，尝试让训练规则从平台期松动迹象转成可持续的传播约束。",
            "next_question": "下一步要把第十八桥并回主核，检验平台期松动是否真正开始向工程规则传播。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十八桥报告",
        "",
        f"- plasticity_rule_alignment_v18: {hm['plasticity_rule_alignment_v18']:.6f}",
        f"- structure_rule_alignment_v18: {hm['structure_rule_alignment_v18']:.6f}",
        f"- topology_training_readiness_v18: {hm['topology_training_readiness_v18']:.6f}",
        f"- topology_training_gap_v18: {hm['topology_training_gap_v18']:.6f}",
        f"- propagation_guard_v18: {hm['propagation_guard_v18']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v18_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
