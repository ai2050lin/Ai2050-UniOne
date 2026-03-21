from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_curriculum_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_spike_3d_long_horizon_plasticity_curriculum_summary() -> dict:
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_reinforcement_20260321" / "summary.json"
    )
    boost = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_boost_20260321" / "summary.json"
    )
    topo_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321" / "summary.json"
    )
    topo_train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )
    ctx_proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )

    hr = reinforce["headline_metrics"]
    hb = boost["headline_metrics"]
    hl = topo_long["headline_metrics"]
    ht = topo_train["headline_metrics"]
    hc = ctx_proto["headline_metrics"]

    curriculum_plasticity_gain = _clip01(
        (
            hr["adaptive_plasticity_gain"]
            + hr["plastic_growth_readiness"]
            + hb["long_horizon_plasticity_boost"]
            + ht["topo_heldout_generalization"] * 0.2
            + hc["heldout_generalization"] * 0.2
        )
        / 3.4
    )
    curriculum_structural_guard = _clip01(
        (
            hr["structural_retention_reinforced"]
            + hl["topo_long_structural_survival"]
            + ht["structural_persistence"]
            + hc["route_split_consistency"]
        )
        / 4.0
    )
    shared_route_guard = _clip01(
        (
            hr["shared_guard_reinforced"]
            + hl["topo_long_shared_survival"]
            + ht["path_reuse_score"]
            + hc["shared_red_consistency"]
        )
        / 4.0
    )
    context_generalization_guard = _clip01(
        (
            hr["contextual_retention_reinforced"]
            + hl["topo_long_context_survival"]
            + hc["context_split_consistency"]
            + ht["route_split_score"]
        )
        / 4.0
    )
    long_horizon_growth_v2 = _clip01(
        (
            curriculum_plasticity_gain
            + curriculum_structural_guard
            + shared_route_guard
            + context_generalization_guard
        )
        / 4.0
    )
    plasticity_curriculum_margin = (
        curriculum_plasticity_gain
        + curriculum_structural_guard
        + shared_route_guard
        + context_generalization_guard
        + long_horizon_growth_v2
    )

    return {
        "headline_metrics": {
            "curriculum_plasticity_gain": curriculum_plasticity_gain,
            "curriculum_structural_guard": curriculum_structural_guard,
            "shared_route_guard": shared_route_guard,
            "context_generalization_guard": context_generalization_guard,
            "long_horizon_growth_v2": long_horizon_growth_v2,
            "plasticity_curriculum_margin": plasticity_curriculum_margin,
        },
        "curriculum_equation": {
            "plasticity_term": "P_curr = mean(P_adapt, R_growth_plus, P_boost, 0.2 * G_topo, 0.2 * G_ctx)",
            "structure_term": "S_curr = mean(R_struct_plus, S_topo_long, S_topo, route_split_consistency)",
            "shared_term": "H_curr = mean(H_guard_plus, H_topo_long, R_topo, shared_red_consistency)",
            "context_term": "C_curr = mean(C_ctx_plus, C_topo_long, context_split_consistency, route_split)",
            "growth_term": "G_curr = mean(P_curr, S_curr, H_curr, C_curr)",
            "system_term": "M_curr = P_curr + S_curr + H_curr + C_curr + G_curr",
        },
        "project_readout": {
            "summary": "长时间尺度三维原型现在开始进入课程式可塑性强化阶段，重点不再只是注入增益，而是把结构保护、共享支路保护和上下文泛化一起绑进增量学习里。",
            "next_question": "下一步要把这组课程式可塑性量并回更大的在线原型，确认更高更新强度下是否还能同时保住结构和持续增长。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 长时间尺度三维脉冲原型课程式可塑性强化报告",
        "",
        f"- curriculum_plasticity_gain: {hm['curriculum_plasticity_gain']:.6f}",
        f"- curriculum_structural_guard: {hm['curriculum_structural_guard']:.6f}",
        f"- shared_route_guard: {hm['shared_route_guard']:.6f}",
        f"- context_generalization_guard: {hm['context_generalization_guard']:.6f}",
        f"- long_horizon_growth_v2: {hm['long_horizon_growth_v2']:.6f}",
        f"- plasticity_curriculum_margin: {hm['plasticity_curriculum_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spike_3d_long_horizon_plasticity_curriculum_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
