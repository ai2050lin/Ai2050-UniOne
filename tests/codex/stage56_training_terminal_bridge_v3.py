from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v3_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v3_summary() -> dict:
    bridge_v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v2_20260321" / "summary.json"
    )
    topo_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321" / "summary.json"
    )
    brain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321" / "summary.json"
    )

    hb = bridge_v2["headline_metrics"]
    ht = topo_long["headline_metrics"]
    hr = brain["headline_metrics"]

    stability_rule_alignment_v3 = _clip01((hb["terminal_bridge_readiness"] + ht["topo_long_retention"] + ht["topo_long_context_survival"]) / 3.0)
    structure_guard_strength_v3 = _clip01((ht["topo_long_structural_survival"] + hr["direct_structure_measure_v3"] + hr["dynamic_structure_balance_v3"]) / 3.0)
    topology_bridge_readiness_v3 = _clip01((stability_rule_alignment_v3 + structure_guard_strength_v3 + hb["topology_rule_alignment"]) / 3.0)
    topology_bridge_gap_v3 = max(0.0, 1.0 - topology_bridge_readiness_v3)

    return {
        "headline_metrics": {
            "stability_rule_alignment_v3": stability_rule_alignment_v3,
            "structure_guard_strength_v3": structure_guard_strength_v3,
            "topology_bridge_readiness_v3": topology_bridge_readiness_v3,
            "topology_bridge_gap_v3": topology_bridge_gap_v3,
        },
        "bridge_equation": {
            "stability_term": "B_stable_v3 = mean(R_train_v2, R_topo_long, C_topo_long)",
            "guard_term": "B_guard_v3 = mean(S_topo_long, D_structure_v3, B_dynamic_v3)",
            "readiness_term": "R_train_v3 = mean(B_stable_v3, B_guard_v3, B_topo)",
            "gap_term": "G_train_v3 = 1 - R_train_v3",
        },
        "project_readout": {
            "summary": "训练终式第三桥把长期三维原型生存率并回训练桥以后，开始更直接地反映结构保持是否足以支持规模化在线学习。",
            "next_question": "下一步要继续提升结构生存率和结构直测强度，否则训练桥会稳定停在中等区而不是强施工区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第三桥报告",
        "",
        f"- stability_rule_alignment_v3: {hm['stability_rule_alignment_v3']:.6f}",
        f"- structure_guard_strength_v3: {hm['structure_guard_strength_v3']:.6f}",
        f"- topology_bridge_readiness_v3: {hm['topology_bridge_readiness_v3']:.6f}",
        f"- topology_bridge_gap_v3: {hm['topology_bridge_gap_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v3_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
