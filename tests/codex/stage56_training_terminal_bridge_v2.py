from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v2_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v2_summary() -> dict:
    train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_rule_20260320" / "summary.json"
    )
    topo = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )
    brain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321" / "summary.json"
    )

    ht = train["headline_metrics"]
    hp = topo["headline_metrics"]
    hh = horizon["headline_metrics"]
    hb = brain["headline_metrics"]

    update_rule_alignment = _clip01((ht["terminal_update_strength"] + hp["topo_train_fit"] + hp["topology_trainable_margin"] / 4.0) / 3.0)
    online_guard_alignment = _clip01((ht["terminal_stability_guard"] + hh["long_horizon_retention"] + hp["structural_persistence"]) / 3.0)
    topology_rule_alignment = _clip01((hp["local_transport_score"] + hp["path_reuse_score"] + hb["direct_route_measure_v3"]) / 3.0)
    terminal_bridge_readiness = _clip01((update_rule_alignment + online_guard_alignment + topology_rule_alignment) / 3.0)
    terminal_bridge_gap = max(0.0, 1.0 - terminal_bridge_readiness)

    return {
        "headline_metrics": {
            "update_rule_alignment": update_rule_alignment,
            "online_guard_alignment": online_guard_alignment,
            "topology_rule_alignment": topology_rule_alignment,
            "terminal_bridge_readiness": terminal_bridge_readiness,
            "terminal_bridge_gap": terminal_bridge_gap,
        },
        "bridge_equation": {
            "update_term": "B_update = mean(U_term, F_topo, M_topo_proto / 4)",
            "guard_term": "B_guard = mean(G_term, R_h_long, S_topo)",
            "topology_term": "B_topo = mean(T_topo, R_topo, D_route_v3)",
            "readiness_term": "R_train_v2 = mean(B_update, B_guard, B_topo)",
            "gap_term": "G_train_v2 = 1 - R_train_v2",
        },
        "project_readout": {
            "summary": "训练终式第二桥把旧训练终式对象、三维拓扑可训练原型、长期在线稳定性和逆向脑编码路由并到同一训练桥里。",
            "next_question": "下一步要把这个训练桥真正放进更大的可训练原型系统，看它能否稳定约束即时学习和结构保持。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二桥报告",
        "",
        f"- update_rule_alignment: {hm['update_rule_alignment']:.6f}",
        f"- online_guard_alignment: {hm['online_guard_alignment']:.6f}",
        f"- topology_rule_alignment: {hm['topology_rule_alignment']:.6f}",
        f"- terminal_bridge_readiness: {hm['terminal_bridge_readiness']:.6f}",
        f"- terminal_bridge_gap: {hm['terminal_bridge_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
