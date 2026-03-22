from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary
from stage105_tensor_level_route_scale_rebuild import build_tensor_level_route_scale_rebuild_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage106_forward_backward_trace_rebuild_20260322"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _monotonic_drop(values: list[float]) -> float:
    decreases = sum(1 for left, right in zip(values, values[1:]) if right <= left)
    return decreases / max(1, len(values) - 1)


@lru_cache(maxsize=1)
def build_forward_backward_trace_rebuild_summary() -> dict:
    language = build_tensor_level_language_projection_rebuild_summary()["headline_metrics"]
    route = build_tensor_level_route_scale_rebuild_summary()["headline_metrics"]
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )["headline_metrics"]
    coupled = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_structure_coupled_validation_20260321" / "summary.json"
    )["headline_metrics"]
    gradient = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_gradient_trajectory_language_probe_20260320" / "summary.json"
    )

    steps = gradient["steps"]
    loss_values = [record["inject_loss"] for record in steps]
    atlas_values = [record["atlas_grad"] for record in steps]
    frontier_values = [record["frontier_grad"] for record in steps]
    boundary_values = [record["boundary_grad"] for record in steps]

    loss_drop_ratio = _clip01((loss_values[0] - loss_values[-1]) / loss_values[0])
    frontier_drop_ratio = _clip01((frontier_values[0] - frontier_values[-1]) / frontier_values[0])
    boundary_drop_ratio = _clip01((boundary_values[0] - boundary_values[-1]) / boundary_values[0])
    atlas_drop_ratio = _clip01((atlas_values[0] - atlas_values[-1]) / atlas_values[0])

    loss_monotonicity = _monotonic_drop(loss_values)
    frontier_monotonicity = _monotonic_drop(frontier_values)
    boundary_monotonicity = _monotonic_drop(boundary_values)
    atlas_monotonicity = _monotonic_drop(atlas_values)

    frontier_dominance = _clip01(
        1.0
        - ((frontier_values[-1] / frontier_values[0]) / ((boundary_values[-1] / boundary_values[0]) + 1e-8)) * 0.25
    )
    boundary_support = _clip01(boundary_drop_ratio * 0.70 + boundary_monotonicity * 0.30)
    atlas_compaction = _clip01(atlas_drop_ratio * 0.70 + atlas_monotonicity * 0.30)

    raw_forward_selectivity = _clip01(
        0.28 * language["reconstructed_route_projection"]
        + 0.24 * route["distributed_network_support"]
        + 0.22 * route["route_structure_coupling_strength"]
        + 0.14 * sparse["sparse_route_activation"]
        + 0.12 * loss_monotonicity
    )
    raw_backward_fidelity = _clip01(
        0.26 * loss_drop_ratio
        + 0.24 * frontier_drop_ratio
        + 0.18 * boundary_drop_ratio
        + 0.16 * frontier_monotonicity
        + 0.16 * boundary_monotonicity
    )
    frontier_boundary_coupling = _clip01(
        0.34 * frontier_dominance
        + 0.26 * boundary_support
        + 0.20 * atlas_compaction
        + 0.20 * route["degradation_tolerance"]
    )
    raw_novelty_binding_capacity = _clip01(
        0.30 * coupled["coupled_novel_gain"]
        + 0.24 * route["degradation_tolerance"]
        + 0.22 * language["reconstructed_bias_transport"]
        + 0.14 * boundary_support
        + 0.10 * sparse["sparse_activation_efficiency"]
    )
    raw_forward_backward_rebuild_score = _clip01(
        0.24 * raw_forward_selectivity
        + 0.24 * raw_backward_fidelity
        + 0.20 * frontier_boundary_coupling
        + 0.18 * raw_novelty_binding_capacity
        + 0.14 * loss_monotonicity
    )

    step_records = []
    for record in steps:
        forward_pressure = _clip01(record["frontier_grad"] / max(frontier_values))
        repair_pressure = _clip01((record["boundary_grad"] + record["atlas_grad"]) / (boundary_values[0] + atlas_values[0]))
        step_records.append(
            {
                "step": record["step"],
                "inject_loss": record["inject_loss"],
                "frontier_grad": record["frontier_grad"],
                "boundary_grad": record["boundary_grad"],
                "atlas_grad": record["atlas_grad"],
                "forward_pressure": forward_pressure,
                "repair_pressure": repair_pressure,
            }
        )

    return {
        "headline_metrics": {
            "loss_drop_ratio": loss_drop_ratio,
            "frontier_drop_ratio": frontier_drop_ratio,
            "boundary_drop_ratio": boundary_drop_ratio,
            "loss_monotonicity": loss_monotonicity,
            "frontier_boundary_coupling": frontier_boundary_coupling,
            "raw_forward_selectivity": raw_forward_selectivity,
            "raw_backward_fidelity": raw_backward_fidelity,
            "raw_novelty_binding_capacity": raw_novelty_binding_capacity,
            "raw_forward_backward_rebuild_score": raw_forward_backward_rebuild_score,
        },
        "step_records": step_records,
        "status": {
            "status_short": (
                "forward_backward_trace_rebuild_ready"
                if raw_forward_backward_rebuild_score >= 0.70 and raw_backward_fidelity >= 0.68
                else "forward_backward_trace_rebuild_transition"
            ),
            "status_label": "前后向计算链已经切换到真实梯度轨迹底座，但仍需更细粒度的张量级训练日志来继续收紧。",
        },
        "project_readout": {
            "summary": "这一步不再直接拼旧的高层闭环分数，而是用真实语言注入轨迹里的 loss、frontier、boundary、atlas 梯度变化，重建前向选路和反向修复的闭环。",
            "next_question": "下一步要继续把这条重建链推进到真实训练批次和更多任务轨迹，检查它是否会在更复杂样本上翻转。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage106 Forward Backward Trace Rebuild",
        "",
        f"- loss_drop_ratio: {hm['loss_drop_ratio']:.6f}",
        f"- frontier_drop_ratio: {hm['frontier_drop_ratio']:.6f}",
        f"- boundary_drop_ratio: {hm['boundary_drop_ratio']:.6f}",
        f"- loss_monotonicity: {hm['loss_monotonicity']:.6f}",
        f"- frontier_boundary_coupling: {hm['frontier_boundary_coupling']:.6f}",
        f"- raw_forward_selectivity: {hm['raw_forward_selectivity']:.6f}",
        f"- raw_backward_fidelity: {hm['raw_backward_fidelity']:.6f}",
        f"- raw_novelty_binding_capacity: {hm['raw_novelty_binding_capacity']:.6f}",
        f"- raw_forward_backward_rebuild_score: {hm['raw_forward_backward_rebuild_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_forward_backward_trace_rebuild_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
