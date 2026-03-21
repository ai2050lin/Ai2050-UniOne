from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_plateau_break_probe_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_plateau_break_probe_summary() -> dict:
    coord = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coordination_stabilization_20260321" / "summary.json"
    )
    mega = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coupled_degradation_validation_20260321" / "summary.json"
    )
    bridge_v16 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v16_20260321" / "summary.json"
    )

    hc = coord["headline_metrics"]
    hm = mega["headline_metrics"]
    hb = bridge_v16["headline_metrics"]

    plateau_structure_guard = _clip01(
        (
            hc["coordinated_structure_guard"]
            + hm["mega_coupled_structure_keep"]
            + hb["structure_rule_alignment_v16"]
            + (1.0 - hm["mega_coupled_collapse_risk"])
        )
        / 4.0
    )
    plateau_context_guard = _clip01(
        (
            hc["coordinated_context_guard"]
            + hm["mega_coupled_context_keep"]
            + hb["coordination_guard_v16"]
            + (1.0 - hm["mega_coupled_forgetting_penalty"])
        )
        / 4.0
    )
    plateau_route_guard = _clip01(
        (
            hc["coordinated_route_guard"]
            + (1.0 - hm["mega_coupled_route_degradation"])
            + hb["coordination_guard_v16"]
            + (1.0 - hc["coordinated_instability_penalty"])
        )
        / 4.0
    )
    plateau_growth_support = _clip01(
        (
            hc["coordinated_growth_support"]
            + hm["mega_coupled_novel_gain"]
            + hb["plasticity_rule_alignment_v16"]
            + hb["topology_training_readiness_v16"]
        )
        / 4.0
    )
    plateau_instability_penalty = _clip01(
        (
            hc["coordinated_instability_penalty"]
            + hm["mega_coupled_forgetting_penalty"]
            + hm["mega_coupled_route_degradation"]
            + hm["mega_coupled_collapse_risk"]
        )
        / 4.0
    )
    plateau_break_readiness = _clip01(
        (
            plateau_structure_guard
            + plateau_context_guard
            + plateau_route_guard
            + plateau_growth_support
            + (1.0 - plateau_instability_penalty)
        )
        / 5.0
    )
    plateau_break_score = _clip01(
        (
            plateau_break_readiness
            + plateau_growth_support
            + plateau_structure_guard
            - plateau_instability_penalty
        )
        / 3.0
    )
    plateau_break_margin = (
        plateau_structure_guard
        + plateau_context_guard
        + plateau_route_guard
        + plateau_growth_support
        + plateau_break_readiness
        + plateau_break_score
        - plateau_instability_penalty
    )

    return {
        "headline_metrics": {
            "plateau_structure_guard": plateau_structure_guard,
            "plateau_context_guard": plateau_context_guard,
            "plateau_route_guard": plateau_route_guard,
            "plateau_growth_support": plateau_growth_support,
            "plateau_instability_penalty": plateau_instability_penalty,
            "plateau_break_readiness": plateau_break_readiness,
            "plateau_break_score": plateau_break_score,
            "plateau_break_margin": plateau_break_margin,
        },
        "plateau_equation": {
            "structure_term": "G_struct_break = mean(G_struct, S_mega, B_struct_v16, 1 - R_collapse_mega)",
            "context_term": "G_ctx_break = mean(G_ctx, C_mega, H_coord_v16, 1 - P_mega)",
            "route_term": "G_route_break = mean(G_route, 1 - R_route_mega, H_coord_v16, 1 - P_coord)",
            "growth_term": "G_growth_break = mean(G_growth, G_mega, B_plastic_v16, R_train_v16)",
            "system_term": "M_break = G_struct_break + G_ctx_break + G_route_break + G_growth_break + R_break - P_break",
        },
        "project_readout": {
            "summary": "更大系统破平台探针开始直接测协同护栏是否足以把系统从平台期里推出来，而不是只看单项指标是否略微改善。",
            "next_question": "下一步要把这组破平台结果并回脑编码直测和训练桥，检验主核是否真正出现突破迹象。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统破平台探针报告",
        "",
        f"- plateau_structure_guard: {hm['plateau_structure_guard']:.6f}",
        f"- plateau_context_guard: {hm['plateau_context_guard']:.6f}",
        f"- plateau_route_guard: {hm['plateau_route_guard']:.6f}",
        f"- plateau_growth_support: {hm['plateau_growth_support']:.6f}",
        f"- plateau_instability_penalty: {hm['plateau_instability_penalty']:.6f}",
        f"- plateau_break_readiness: {hm['plateau_break_readiness']:.6f}",
        f"- plateau_break_score: {hm['plateau_break_score']:.6f}",
        f"- plateau_break_margin: {hm['plateau_break_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_plateau_break_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
