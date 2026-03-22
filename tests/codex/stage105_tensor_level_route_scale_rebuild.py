from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage105_tensor_level_route_scale_rebuild_20260322"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def build_tensor_level_route_scale_rebuild_summary() -> dict:
    language = build_tensor_level_language_projection_rebuild_summary()["headline_metrics"]
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )["headline_metrics"]
    coupled = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_structure_coupled_validation_20260321" / "summary.json"
    )["headline_metrics"]
    degradation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_degradation_probe_20260321" / "summary.json"
    )["headline_metrics"]

    local_anchor_support = _clip01(
        0.28 * sparse["sparse_seed_activation"]
        + 0.22 * sparse["sparse_feature_activation"]
        + 0.18 * sparse["sparse_structure_activation"]
        + 0.16 * (1.0 - degradation["route_degradation_risk"])
        + 0.16 * (1.0 - coupled["coupled_failure_risk"])
    )
    mesoscopic_bundle_support = _clip01(
        0.28 * coupled["coupled_route_keep"]
        + 0.24 * coupled["coupled_structure_keep"]
        + 0.18 * degradation["structure_resilience"]
        + 0.14 * coupled["coupled_context_keep"]
        + 0.16 * sparse["sparse_structure_activation"]
    )
    distributed_network_support = _clip01(
        0.22 * coupled["coupled_context_keep"]
        + 0.20 * coupled["coupled_novel_gain"]
        + 0.18 * degradation["route_resilience"]
        + 0.14 * degradation["true_scale_reinforced_readiness"]
        + 0.14 * coupled["coupled_readiness"]
        + 0.12 * language["reconstructed_route_projection"]
    )
    route_structure_coupling_strength = _clip01(
        0.30 * coupled["coupled_route_keep"]
        + 0.28 * coupled["coupled_structure_keep"]
        + 0.22 * coupled["coupled_context_keep"]
        + 0.20 * language["reconstructed_context_gate_coherence"]
    )
    degradation_tolerance = _clip01(
        1.0
        - (
            0.36 * degradation["route_degradation_risk"]
            + 0.28 * degradation["structure_phase_shift_risk"]
            + 0.22 * coupled["coupled_failure_risk"]
            + 0.14 * coupled["coupled_forgetting_penalty"]
        )
    )
    dominant_support = max(local_anchor_support, mesoscopic_bundle_support, distributed_network_support)
    if dominant_support == distributed_network_support:
        dominant_scale_name = "distributed_network"
    elif dominant_support == mesoscopic_bundle_support:
        dominant_scale_name = "mesoscopic_bundle"
    else:
        dominant_scale_name = "local_anchor"

    route_scale_margin = distributed_network_support - max(local_anchor_support, mesoscopic_bundle_support)
    reconstructed_route_scale_score = _clip01(
        0.28 * distributed_network_support
        + 0.20 * route_structure_coupling_strength
        + 0.20 * degradation_tolerance
        + 0.16 * language["reconstructed_route_projection"]
        + 0.16 * (0.5 + 0.5 * _clip01(route_scale_margin + 0.20))
    )

    return {
        "headline_metrics": {
            "local_anchor_support": local_anchor_support,
            "mesoscopic_bundle_support": mesoscopic_bundle_support,
            "distributed_network_support": distributed_network_support,
            "route_structure_coupling_strength": route_structure_coupling_strength,
            "degradation_tolerance": degradation_tolerance,
            "route_scale_margin": route_scale_margin,
            "reconstructed_route_scale_score": reconstructed_route_scale_score,
            "dominant_scale_name": dominant_scale_name,
        },
        "status": {
            "status_short": (
                "tensor_level_route_scale_rebuild_ready"
                if reconstructed_route_scale_score >= 0.72 and dominant_scale_name == "distributed_network"
                else "tensor_level_route_scale_rebuild_transition"
            ),
            "status_label": "路由尺度链已改用真实探针摘要重建，当前主导尺度仍稳定落在分布式网络。",
        },
        "project_readout": {
            "summary": "这一步不再直接复用旧的尺度支撑分数，而是用稀疏激活、真实规模路由退化和耦合验证结果重建路由尺度链。",
            "next_question": "下一步要把这条尺度链和真实梯度轨迹接起来，形成更底层的前后向闭环证据。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage105 Tensor Level Route Scale Rebuild",
        "",
        f"- local_anchor_support: {hm['local_anchor_support']:.6f}",
        f"- mesoscopic_bundle_support: {hm['mesoscopic_bundle_support']:.6f}",
        f"- distributed_network_support: {hm['distributed_network_support']:.6f}",
        f"- route_structure_coupling_strength: {hm['route_structure_coupling_strength']:.6f}",
        f"- degradation_tolerance: {hm['degradation_tolerance']:.6f}",
        f"- route_scale_margin: {hm['route_scale_margin']:.6f}",
        f"- reconstructed_route_scale_score: {hm['reconstructed_route_scale_score']:.6f}",
        f"- dominant_scale_name: {hm['dominant_scale_name']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_tensor_level_route_scale_rebuild_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
