from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_reinforcement_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_spike_3d_long_horizon_plasticity_reinforcement_summary() -> dict:
    boost = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_boost_20260321" / "summary.json"
    )
    topo_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321" / "summary.json"
    )
    topo_train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )
    online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_rollback_probe_20260321" / "summary.json"
    )

    hb = boost["headline_metrics"]
    ht = topo_long["headline_metrics"]
    hp = topo_train["headline_metrics"]
    hh = horizon["headline_metrics"]
    ho = online["headline_metrics"]

    adaptive_plasticity_gain = _clip01(
        (
            hb["long_horizon_plasticity_boost"]
            + hb["injected_plasticity_gain"]
            + hh["long_horizon_plasticity"]
            + ho["online_gain"]
            + hp["topo_heldout_generalization"] * 0.25
        )
        / 4.25
    )
    structural_retention_reinforced = _clip01(
        (
            hb["structural_plasticity_balance"]
            + ht["topo_long_structural_survival"]
            + hp["structural_persistence"]
            + hh["structural_survival"]
        )
        / 4.0
    )
    shared_guard_reinforced = _clip01(
        (
            hb["shared_guard_after_boost"]
            + ht["topo_long_shared_survival"]
            + hh["shared_fiber_survival"]
            + hp["path_reuse_score"]
        )
        / 4.0
    )
    contextual_retention_reinforced = _clip01(
        (
            ht["topo_long_context_survival"]
            + hh["contextual_survival"]
            + ho["context_split_retention"]
            + hp["route_split_score"]
        )
        / 4.0
    )
    plastic_growth_readiness = _clip01(
        (
            adaptive_plasticity_gain
            + structural_retention_reinforced
            + shared_guard_reinforced
            + contextual_retention_reinforced
        )
        / 4.0
    )
    plasticity_reinforced_margin = (
        adaptive_plasticity_gain
        + structural_retention_reinforced
        + shared_guard_reinforced
        + contextual_retention_reinforced
        + plastic_growth_readiness
    )

    return {
        "headline_metrics": {
            "adaptive_plasticity_gain": adaptive_plasticity_gain,
            "structural_retention_reinforced": structural_retention_reinforced,
            "shared_guard_reinforced": shared_guard_reinforced,
            "contextual_retention_reinforced": contextual_retention_reinforced,
            "plastic_growth_readiness": plastic_growth_readiness,
            "plasticity_reinforced_margin": plasticity_reinforced_margin,
        },
        "reinforcement_equation": {
            "adaptive_term": "P_adapt = mean(P_boost, P_inject, G_h, G_online, 0.25 * G_topo)",
            "structure_term": "R_struct_plus = mean(S_boost, S_topo_long, S_topo, H_structure)",
            "guard_term": "H_guard_plus = mean(H_boost, H_topo_long, H_fiber_long, R_topo)",
            "context_term": "C_ctx_plus = mean(C_topo_long, H_context, R_context, route_split)",
            "readiness_term": "R_growth_plus = mean(P_adapt, R_struct_plus, H_guard_plus, C_ctx_plus)",
            "system_term": "M_plasticity_plus = P_adapt + R_struct_plus + H_guard_plus + C_ctx_plus + R_growth_plus",
        },
        "project_readout": {
            "summary": "长时间尺度三维原型已经从单纯保结构，推进到了可塑性强化阶段，但当前最真实的难点仍然是如何把增量学习能力继续抬高，同时不打掉共享属性支路和结构保持。",
            "next_question": "下一步要把这组可塑性强化量并回训练桥，看训练规则能否同时支撑长期保结构和持续吃进新知识。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 长时间尺度三维脉冲原型可塑性强化报告",
        "",
        f"- adaptive_plasticity_gain: {hm['adaptive_plasticity_gain']:.6f}",
        f"- structural_retention_reinforced: {hm['structural_retention_reinforced']:.6f}",
        f"- shared_guard_reinforced: {hm['shared_guard_reinforced']:.6f}",
        f"- contextual_retention_reinforced: {hm['contextual_retention_reinforced']:.6f}",
        f"- plastic_growth_readiness: {hm['plastic_growth_readiness']:.6f}",
        f"- plasticity_reinforced_margin: {hm['plasticity_reinforced_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spike_3d_long_horizon_plasticity_reinforcement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
