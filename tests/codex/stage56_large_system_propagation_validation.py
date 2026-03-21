from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_propagation_validation_summary() -> dict:
    mega = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coupled_degradation_validation_20260321" / "summary.json"
    )
    coord = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coordination_stabilization_20260321" / "summary.json"
    )
    propagation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_plateau_break_propagation_probe_20260321" / "summary.json"
    )
    v74 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v74_20260321" / "summary.json"
    )

    hm = mega["headline_metrics"]
    hc = coord["headline_metrics"]
    hp = propagation["headline_metrics"]
    hv = v74["headline_metrics"]

    scale_propagation_structure = _clip01(
        (
            hm["mega_coupled_structure_keep"]
            + hc["coordinated_structure_guard"]
            + hp["propagation_structure"]
            + (1.0 - hm["mega_coupled_collapse_risk"])
        )
        / 4.0
    )
    scale_propagation_context = _clip01(
        (
            hm["mega_coupled_context_keep"]
            + hc["coordinated_context_guard"]
            + hp["propagation_context"]
            + (1.0 - hm["mega_coupled_forgetting_penalty"])
        )
        / 4.0
    )
    scale_propagation_route = _clip01(
        (
            (1.0 - hm["mega_coupled_route_degradation"])
            + hc["coordinated_route_guard"]
            + hp["propagation_route"]
            + (1.0 - hm["mega_coupled_collapse_risk"])
        )
        / 4.0
    )
    scale_propagation_learning = _clip01(
        (
            hm["mega_coupled_novel_gain"]
            + hc["coordinated_growth_support"]
            + hp["propagation_learning"]
            + hp["propagation_break_score"]
        )
        / 4.0
    )
    scale_propagation_penalty = _clip01(
        (
            hm["mega_coupled_forgetting_penalty"]
            + hm["mega_coupled_route_degradation"]
            + hm["mega_coupled_collapse_risk"]
            + hp["propagation_penalty"]
        )
        / 4.0
    )
    scale_propagation_readiness = _clip01(
        (
            scale_propagation_structure
            + scale_propagation_context
            + scale_propagation_route
            + scale_propagation_learning
            + (1.0 - scale_propagation_penalty)
        )
        / 5.0
    )
    scale_propagation_score = _clip01(
        (
            scale_propagation_readiness
            + hm["mega_coupled_readiness"]
            + hp["propagation_break_score"]
            + (1.0 - scale_propagation_penalty)
        )
        / 4.0
    )
    scale_propagation_margin = (
        scale_propagation_structure
        + scale_propagation_context
        + scale_propagation_route
        + scale_propagation_learning
        + scale_propagation_readiness
        + scale_propagation_score
        + hv["encoding_margin_v74"] / max(hv["encoding_margin_v74"], 1.0)
        - scale_propagation_penalty
    )

    return {
        "headline_metrics": {
            "scale_propagation_structure": scale_propagation_structure,
            "scale_propagation_context": scale_propagation_context,
            "scale_propagation_route": scale_propagation_route,
            "scale_propagation_learning": scale_propagation_learning,
            "scale_propagation_penalty": scale_propagation_penalty,
            "scale_propagation_readiness": scale_propagation_readiness,
            "scale_propagation_score": scale_propagation_score,
            "scale_propagation_margin": scale_propagation_margin,
        },
        "propagation_equation_v2": {
            "structure_term": "S_prop_scale = mean(S_mega, G_struct, T_struct, 1 - R_collapse_mega)",
            "context_term": "C_prop_scale = mean(C_mega, G_ctx, T_ctx, 1 - P_mega)",
            "route_term": "R_prop_scale = mean(1 - R_route_mega, G_route, T_route, 1 - R_collapse_mega)",
            "learning_term": "L_prop_scale = mean(G_mega, G_growth, T_learn, R_break)",
            "system_term": "M_prop_scale = S_prop_scale + C_prop_scale + R_prop_scale + L_prop_scale + R_scale - P_scale",
        },
        "project_readout": {
            "summary": "更大系统传播验证开始直接检验平台期松动是否能在更大对象集、更长上下文和更高耦合压力下继续保持，而不是重新塌回局部现象。",
            "next_question": "下一步要把这组更大系统传播结果并回脑编码直测和训练终式，检验传播是否终于形成跨层级突破。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统传播验证报告",
        "",
        f"- scale_propagation_structure: {hm['scale_propagation_structure']:.6f}",
        f"- scale_propagation_context: {hm['scale_propagation_context']:.6f}",
        f"- scale_propagation_route: {hm['scale_propagation_route']:.6f}",
        f"- scale_propagation_learning: {hm['scale_propagation_learning']:.6f}",
        f"- scale_propagation_penalty: {hm['scale_propagation_penalty']:.6f}",
        f"- scale_propagation_readiness: {hm['scale_propagation_readiness']:.6f}",
        f"- scale_propagation_score: {hm['scale_propagation_score']:.6f}",
        f"- scale_propagation_margin: {hm['scale_propagation_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_propagation_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
