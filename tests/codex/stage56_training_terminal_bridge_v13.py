from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v13_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v13_summary() -> dict:
    v12 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v12_20260321" / "summary.json"
    )
    route_probe = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_degradation_probe_20260321" / "summary.json"
    )
    brain_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v7_20260321" / "summary.json"
    )

    hv = v12["headline_metrics"]
    hr = route_probe["headline_metrics"]
    hb = brain_v7["headline_metrics"]

    plasticity_rule_alignment_v13 = _clip01(
        (
            hv["plasticity_rule_alignment_v12"]
            + hr["route_resilience"]
            + (1.0 - hr["route_degradation_risk"])
            + hb["direct_feature_measure_v7"]
            + (1.0 - hb["direct_brain_gap_v7"])
        )
        / 5.0
    )
    structure_rule_alignment_v13 = _clip01(
        (
            hv["structure_rule_alignment_v12"]
            + hr["structure_resilience"]
            + (1.0 - hr["structure_phase_shift_risk"])
            + hb["direct_structure_measure_v7"]
            + hb["direct_route_measure_v7"]
        )
        / 5.0
    )
    topology_training_readiness_v13 = _clip01(
        (
            hv["topology_training_readiness_v12"]
            + plasticity_rule_alignment_v13
            + structure_rule_alignment_v13
            + hr["true_scale_reinforced_readiness"]
            + hb["direct_route_alignment_v7"]
            + (1.0 - hr["route_degradation_risk"])
        )
        / 6.0
    )
    topology_training_gap_v13 = max(0.0, 1.0 - topology_training_readiness_v13)
    route_guard_v13 = _clip01(
        (
            hr["route_resilience"]
            + hr["structure_resilience"]
            + topology_training_readiness_v13
            + hb["direct_route_measure_v7"]
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v13": plasticity_rule_alignment_v13,
            "structure_rule_alignment_v13": structure_rule_alignment_v13,
            "topology_training_readiness_v13": topology_training_readiness_v13,
            "topology_training_gap_v13": topology_training_gap_v13,
            "route_guard_v13": route_guard_v13,
        },
        "bridge_equation_v13": {
            "plasticity_term": "B_plastic_v13 = mean(B_plastic_v12, H_route, 1 - R_route, D_feature_v7, 1 - G_brain_v7)",
            "structure_term": "B_struct_v13 = mean(B_struct_v12, H_struct, 1 - R_phase, D_structure_v7, D_route_v7)",
            "readiness_term": "R_train_v13 = mean(R_train_v12, B_plastic_v13, B_struct_v13, A_route, D_route_align_v7, 1 - R_route)",
            "gap_term": "G_train_v13 = 1 - R_train_v13",
            "guard_term": "H_route_v13 = mean(H_route, H_struct, R_train_v13, D_route_v7)",
        },
        "project_readout": {
            "summary": "训练终式第十三桥开始显式吸收路由退化和结构相变风险，使训练规则不只面对一般塌缩，还开始面向真实大规模在线系统里的路由退化。",
            "next_question": "下一步要把第十三桥并回主核，检验主核在更真实路由退化压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十三桥报告",
        "",
        f"- plasticity_rule_alignment_v13: {hm['plasticity_rule_alignment_v13']:.6f}",
        f"- structure_rule_alignment_v13: {hm['structure_rule_alignment_v13']:.6f}",
        f"- topology_training_readiness_v13: {hm['topology_training_readiness_v13']:.6f}",
        f"- topology_training_gap_v13: {hm['topology_training_gap_v13']:.6f}",
        f"- route_guard_v13: {hm['route_guard_v13']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v13_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
