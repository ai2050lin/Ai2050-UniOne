from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_coordination_stabilization_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_coordination_stabilization_summary() -> dict:
    mega = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coupled_degradation_validation_20260321" / "summary.json"
    )
    coupled = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_structure_coupled_validation_20260321" / "summary.json"
    )
    bridge_v15 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v15_20260321" / "summary.json"
    )

    hm = mega["headline_metrics"]
    hc = coupled["headline_metrics"]
    hb = bridge_v15["headline_metrics"]

    coordinated_structure_guard = _clip01(
        (
            hm["mega_coupled_structure_keep"]
            + hc["coupled_structure_keep"]
            + hb["structure_rule_alignment_v15"]
            + (1.0 - hm["mega_coupled_collapse_risk"])
        )
        / 4.0
    )
    coordinated_context_guard = _clip01(
        (
            hm["mega_coupled_context_keep"]
            + hc["coupled_context_keep"]
            + hb["mega_guard_v15"]
            + (1.0 - hm["mega_coupled_forgetting_penalty"])
        )
        / 4.0
    )
    coordinated_route_guard = _clip01(
        (
            (1.0 - hm["mega_coupled_route_degradation"])
            + hc["coupled_route_keep"]
            + hb["mega_guard_v15"]
            + (1.0 - hc["coupled_failure_risk"])
        )
        / 4.0
    )
    coordinated_growth_support = _clip01(
        (
            hm["mega_coupled_novel_gain"]
            + hc["coupled_novel_gain"]
            + hb["plasticity_rule_alignment_v15"]
            + hb["topology_training_readiness_v15"]
        )
        / 4.0
    )
    coordinated_instability_penalty = _clip01(
        (
            hm["mega_coupled_forgetting_penalty"]
            + hm["mega_coupled_route_degradation"]
            + hm["mega_coupled_collapse_risk"]
            + hc["coupled_failure_risk"]
        )
        / 4.0
    )
    coordinated_readiness = _clip01(
        (
            coordinated_structure_guard
            + coordinated_context_guard
            + coordinated_route_guard
            + coordinated_growth_support
            + (1.0 - coordinated_instability_penalty)
        )
        / 5.0
    )
    coordinated_margin = (
        coordinated_structure_guard
        + coordinated_context_guard
        + coordinated_route_guard
        + coordinated_growth_support
        + coordinated_readiness
        - coordinated_instability_penalty
    )

    return {
        "headline_metrics": {
            "coordinated_structure_guard": coordinated_structure_guard,
            "coordinated_context_guard": coordinated_context_guard,
            "coordinated_route_guard": coordinated_route_guard,
            "coordinated_growth_support": coordinated_growth_support,
            "coordinated_instability_penalty": coordinated_instability_penalty,
            "coordinated_readiness": coordinated_readiness,
            "coordinated_margin": coordinated_margin,
        },
        "coordination_equation": {
            "structure_term": "G_struct = mean(S_mega, K_struct, B_struct_v15, 1 - R_collapse_mega)",
            "context_term": "G_ctx = mean(C_mega, K_ctx, H_mega_v15, 1 - P_mega)",
            "route_term": "G_route = mean(1 - R_route_mega, K_route, H_mega_v15, 1 - R_fail)",
            "growth_term": "G_growth = mean(G_mega, G_coupled, B_plastic_v15, R_train_v15)",
            "system_term": "M_coord = G_struct + G_ctx + G_route + G_growth + R_coord - P_coord",
        },
        "project_readout": {
            "summary": "更大系统协同稳定化开始专门处理结构、上下文和路由三条线一起变脆的问题，使主线第一次从单项补强转向协同护栏。",
            "next_question": "下一步要把这组协同稳定化结果并回脑编码直测和训练桥，检验主核能否突破当前的平台期。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统协同稳定化报告",
        "",
        f"- coordinated_structure_guard: {hm['coordinated_structure_guard']:.6f}",
        f"- coordinated_context_guard: {hm['coordinated_context_guard']:.6f}",
        f"- coordinated_route_guard: {hm['coordinated_route_guard']:.6f}",
        f"- coordinated_growth_support: {hm['coordinated_growth_support']:.6f}",
        f"- coordinated_instability_penalty: {hm['coordinated_instability_penalty']:.6f}",
        f"- coordinated_readiness: {hm['coordinated_readiness']:.6f}",
        f"- coordinated_margin: {hm['coordinated_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_coordination_stabilization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
