from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_plateau_break_propagation_probe_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_plateau_break_propagation_probe_summary() -> dict:
    plateau = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_plateau_break_probe_20260321" / "summary.json"
    )
    coord = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coordination_stabilization_20260321" / "summary.json"
    )
    bridge_v17 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v17_20260321" / "summary.json"
    )

    hp = plateau["headline_metrics"]
    hc = coord["headline_metrics"]
    hb = bridge_v17["headline_metrics"]

    propagation_structure = _clip01(
        (
            hp["plateau_structure_guard"]
            + hc["coordinated_structure_guard"]
            + hb["structure_rule_alignment_v17"]
            + hb["plateau_guard_v17"]
        )
        / 4.0
    )
    propagation_context = _clip01(
        (
            hp["plateau_context_guard"]
            + hc["coordinated_context_guard"]
            + hb["plateau_guard_v17"]
            + (1.0 - hp["plateau_instability_penalty"])
        )
        / 4.0
    )
    propagation_route = _clip01(
        (
            hp["plateau_route_guard"]
            + hc["coordinated_route_guard"]
            + hb["plateau_guard_v17"]
            + (1.0 - hp["plateau_instability_penalty"])
        )
        / 4.0
    )
    propagation_learning = _clip01(
        (
            hp["plateau_growth_support"]
            + hb["plasticity_rule_alignment_v17"]
            + hb["topology_training_readiness_v17"]
            + hp["plateau_break_readiness"]
        )
        / 4.0
    )
    propagation_penalty = _clip01(
        (
            hp["plateau_instability_penalty"]
            + (1.0 - hb["plateau_guard_v17"])
            + hb["topology_training_gap_v17"]
        )
        / 3.0
    )
    propagation_readiness = _clip01(
        (
            propagation_structure
            + propagation_context
            + propagation_route
            + propagation_learning
            + (1.0 - propagation_penalty)
        )
        / 5.0
    )
    propagation_break_score = _clip01(
        (
            propagation_readiness
            + propagation_learning
            + propagation_structure
            + propagation_route
            - propagation_penalty
        )
        / 4.0
    )
    propagation_margin = (
        propagation_structure
        + propagation_context
        + propagation_route
        + propagation_learning
        + propagation_readiness
        + propagation_break_score
        - propagation_penalty
    )

    return {
        "headline_metrics": {
            "propagation_structure": propagation_structure,
            "propagation_context": propagation_context,
            "propagation_route": propagation_route,
            "propagation_learning": propagation_learning,
            "propagation_penalty": propagation_penalty,
            "propagation_readiness": propagation_readiness,
            "propagation_break_score": propagation_break_score,
            "propagation_margin": propagation_margin,
        },
        "propagation_equation": {
            "structure_term": "T_struct = mean(G_struct_break, G_struct, B_struct_v17, H_break_v17)",
            "context_term": "T_ctx = mean(G_ctx_break, G_ctx, H_break_v17, 1 - P_break)",
            "route_term": "T_route = mean(G_route_break, G_route, H_break_v17, 1 - P_break)",
            "learning_term": "T_learn = mean(G_growth_break, B_plastic_v17, R_train_v17, R_break)",
            "system_term": "M_prop = T_struct + T_ctx + T_route + T_learn + R_prop - P_prop",
        },
        "project_readout": {
            "summary": "破平台传播探针开始直接检验平台期松动是否能传导到结构、上下文、路由和学习四条线，而不是只停在局部信号层。",
            "next_question": "下一步要把这组传播结果并回脑编码直测和训练桥，检验主核是否真的出现传播级突破。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 破平台传播探针报告",
        "",
        f"- propagation_structure: {hm['propagation_structure']:.6f}",
        f"- propagation_context: {hm['propagation_context']:.6f}",
        f"- propagation_route: {hm['propagation_route']:.6f}",
        f"- propagation_learning: {hm['propagation_learning']:.6f}",
        f"- propagation_penalty: {hm['propagation_penalty']:.6f}",
        f"- propagation_readiness: {hm['propagation_readiness']:.6f}",
        f"- propagation_break_score: {hm['propagation_break_score']:.6f}",
        f"- propagation_margin: {hm['propagation_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_plateau_break_propagation_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
