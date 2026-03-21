from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_attenuation_probe_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_propagation_attenuation_probe_summary() -> dict:
    local_prop = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_plateau_break_propagation_probe_20260321" / "summary.json"
    )
    scale_prop = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_validation_20260321" / "summary.json"
    )
    bridge_v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v19_20260321" / "summary.json"
    )

    hl = local_prop["headline_metrics"]
    hs = scale_prop["headline_metrics"]
    hb = bridge_v19["headline_metrics"]

    attenuation_structure = _clip01(hl["propagation_structure"] - hs["scale_propagation_structure"])
    attenuation_context = _clip01(hl["propagation_context"] - hs["scale_propagation_context"] + 0.01)
    attenuation_route = _clip01(hl["propagation_route"] - hs["scale_propagation_route"])
    attenuation_learning = _clip01(hl["propagation_learning"] - hs["scale_propagation_learning"])
    attenuation_penalty = _clip01(
        (
            attenuation_structure
            + attenuation_context
            + attenuation_route
            + attenuation_learning
            + hs["scale_propagation_penalty"]
        )
        / 5.0
    )
    anti_attenuation_readiness = _clip01(
        (
            hb["plasticity_rule_alignment_v19"]
            + hb["structure_rule_alignment_v19"]
            + hs["scale_propagation_readiness"]
            + (1.0 - attenuation_penalty)
        )
        / 4.0
    )
    attenuation_gap = _clip01(1.0 - anti_attenuation_readiness)
    anti_attenuation_margin = (
        anti_attenuation_readiness
        + (1.0 - attenuation_structure)
        + (1.0 - attenuation_context)
        + (1.0 - attenuation_route)
        + (1.0 - attenuation_learning)
        - attenuation_penalty
    )

    return {
        "headline_metrics": {
            "attenuation_structure": attenuation_structure,
            "attenuation_context": attenuation_context,
            "attenuation_route": attenuation_route,
            "attenuation_learning": attenuation_learning,
            "attenuation_penalty": attenuation_penalty,
            "anti_attenuation_readiness": anti_attenuation_readiness,
            "attenuation_gap": attenuation_gap,
            "anti_attenuation_margin": anti_attenuation_margin,
        },
        "attenuation_equation": {
            "structure_term": "A_struct = propagation_structure - scale_propagation_structure",
            "context_term": "A_ctx = propagation_context - scale_propagation_context",
            "route_term": "A_route = propagation_route - scale_propagation_route",
            "learning_term": "A_learn = propagation_learning - scale_propagation_learning",
            "system_term": "M_anti_att = R_anti_att + (1 - A_struct) + (1 - A_ctx) + (1 - A_route) + (1 - A_learn) - P_att",
        },
        "project_readout": {
            "summary": "更大系统传播衰减探针开始直接测平台期松动在规模化场景下的衰减量，并评估现有护栏是否足以对抗这种衰减。",
            "next_question": "下一步要把这组衰减结果并回脑编码直测和训练终式，检验系统能否开始从衰减走向补偿。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统传播衰减探针报告",
        "",
        f"- attenuation_structure: {hm['attenuation_structure']:.6f}",
        f"- attenuation_context: {hm['attenuation_context']:.6f}",
        f"- attenuation_route: {hm['attenuation_route']:.6f}",
        f"- attenuation_learning: {hm['attenuation_learning']:.6f}",
        f"- attenuation_penalty: {hm['attenuation_penalty']:.6f}",
        f"- anti_attenuation_readiness: {hm['anti_attenuation_readiness']:.6f}",
        f"- attenuation_gap: {hm['attenuation_gap']:.6f}",
        f"- anti_attenuation_margin: {hm['anti_attenuation_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_propagation_attenuation_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
