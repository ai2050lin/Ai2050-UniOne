from __future__ import annotations

import json
import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage90_independent_observation_planes_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage73_falsifiability_boundary_hardening import build_falsifiability_boundary_hardening_summary
from stage77_brain_grounded_route_scaling import build_brain_grounded_route_scaling_summary
from stage78_distributed_route_native_observability import build_distributed_route_native_observability_summary
from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary
from stage81_forward_backward_unification import build_forward_backward_unification_summary
from stage82_novelty_generalization_repair import build_novelty_generalization_repair_summary
from stage84_falsifiable_computation_core import build_falsifiable_computation_core_summary
from stage87_evidence_independence_audit import build_evidence_independence_audit_summary
from stage88_external_counterexample_expansion import build_external_counterexample_expansion_summary
from stage89_law_margin_separation import build_law_margin_separation_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@lru_cache(maxsize=1)
def build_independent_observation_planes_summary() -> dict:
    language = build_language_projection_covariance_summary()["headline_metrics"]
    boundary = build_falsifiability_boundary_hardening_summary()["headline_metrics"]
    brain_scale = build_brain_grounded_route_scaling_summary()["headline_metrics"]
    brain_obs = build_distributed_route_native_observability_summary()["headline_metrics"]
    route = build_route_conflict_native_measure_summary()["headline_metrics"]
    loop = build_forward_backward_unification_summary()["headline_metrics"]
    novelty = build_novelty_generalization_repair_summary()["headline_metrics"]
    falsifiable = build_falsifiable_computation_core_summary()["headline_metrics"]
    audit = build_evidence_independence_audit_summary()["headline_metrics"]
    external = build_external_counterexample_expansion_summary()["headline_metrics"]
    law_margin = build_law_margin_separation_summary()["headline_metrics"]

    plane_records = [
        {
            "name": "language_plane",
            "sources": {"stage72"},
            "state_variables": {"a", "r", "f", "g", "q", "b"},
            "anchors": {
                "context_covariance_stability",
                "bias_gate_transport",
                "route_conditioned_projection",
                "projection_gap",
            },
            "signal_strength": _clip01(
                0.28 * language["context_covariance_stability"]
                + 0.24 * language["bias_gate_transport"]
                + 0.24 * language["route_conditioned_projection"]
                + 0.24 * (1.0 - language["projection_gap"])
            ),
        },
        {
            "name": "brain_plane",
            "sources": {"stage77", "stage78"},
            "state_variables": {"a", "r", "f", "g", "p", "h", "m", "c"},
            "anchors": {
                "distributed_network_support",
                "route_scale_grounding_score",
                "brain_constrained_repair_score",
                "field_proxy_gap",
            },
            "signal_strength": _clip01(
                0.28 * brain_scale["distributed_network_support"]
                + 0.24 * brain_scale["route_scale_grounding_score"]
                + 0.24 * brain_scale["brain_constrained_repair_score"]
                + 0.24 * (1.0 - brain_obs["field_proxy_gap"])
            ),
        },
        {
            "name": "intelligence_plane",
            "sources": {"stage79", "stage81", "stage82", "stage89"},
            "state_variables": {"g", "q", "f", "p", "h", "m", "c"},
            "anchors": {
                "route_computation_closure_score",
                "forward_backward_unification_score",
                "best_repaired_novelty_score",
                "law_margin_separation_score",
            },
            "signal_strength": _clip01(
                0.24 * route["route_computation_closure_score"]
                + 0.24 * loop["forward_backward_unification_score"]
                + 0.20 * novelty["best_repaired_novelty_score"]
                + 0.18 * law_margin["law_margin_separation_score"]
                + 0.14 * (1.0 - novelty["best_failure_after"])
            ),
        },
        {
            "name": "falsification_plane",
            "sources": {"stage73", "stage84", "stage88"},
            "state_variables": {"g", "q", "p", "h", "m", "c"},
            "anchors": {
                "falsifiability_boundary_hardening_score",
                "falsifiable_computation_core_score",
                "external_counterexample_expansion_score",
                "hardest_counterexample_intensity",
            },
            "signal_strength": _clip01(
                0.24 * boundary["falsifiability_boundary_hardening_score"]
                + 0.24 * falsifiable["falsifiable_computation_core_score"]
                + 0.20 * external["external_counterexample_expansion_score"]
                + 0.16 * boundary["shared_state_rejection_power"]
                + 0.16 * (1.0 - falsifiable["hardest_counterexample_intensity"])
            ),
        },
    ]

    anchor_overlap_values = []
    variable_overlap_values = []
    source_overlap_values = []
    overlap_matrix = []
    for left, right in combinations(plane_records, 2):
        anchor_overlap = _jaccard(left["anchors"], right["anchors"])
        variable_overlap = _jaccard(left["state_variables"], right["state_variables"])
        source_overlap = _jaccard(left["sources"], right["sources"])
        anchor_overlap_values.append(anchor_overlap)
        variable_overlap_values.append(variable_overlap)
        source_overlap_values.append(source_overlap)
        overlap_matrix.append(
            {
                "pair": f"{left['name']}__{right['name']}",
                "anchor_overlap": anchor_overlap,
                "variable_overlap": variable_overlap,
                "source_overlap": source_overlap,
            }
        )

    plane_signal_mean = sum(record["signal_strength"] for record in plane_records) / len(plane_records)
    surface_anchor_independence = 1.0 - (sum(anchor_overlap_values) / len(anchor_overlap_values))
    variable_coupling_overlap = sum(variable_overlap_values) / len(variable_overlap_values)
    source_plane_separation = 1.0 - (sum(source_overlap_values) / len(source_overlap_values))

    all_anchors = [anchor for record in plane_records for anchor in record["anchors"]]
    exclusive_anchor_ratio = len(set(all_anchors)) / len(all_anchors)

    backfeed_risk_after_split = _clip01(
        audit["summary_backfeed_risk"]
        - 0.22 * surface_anchor_independence
        - 0.12 * source_plane_separation
        - 0.10 * plane_signal_mean
    )
    independent_observation_planes_score = _clip01(
        0.26 * surface_anchor_independence
        + 0.22 * source_plane_separation
        + 0.20 * exclusive_anchor_ratio
        + 0.18 * plane_signal_mean
        + 0.14 * (1.0 - variable_coupling_overlap)
    )

    return {
        "headline_metrics": {
            "plane_signal_mean": plane_signal_mean,
            "surface_anchor_independence": surface_anchor_independence,
            "source_plane_separation": source_plane_separation,
            "exclusive_anchor_ratio": exclusive_anchor_ratio,
            "variable_coupling_overlap": variable_coupling_overlap,
            "backfeed_risk_after_split": backfeed_risk_after_split,
            "independent_observation_planes_score": independent_observation_planes_score,
        },
        "plane_records": [
            {
                "name": record["name"],
                "sources": sorted(record["sources"]),
                "state_variables": sorted(record["state_variables"]),
                "anchors": sorted(record["anchors"]),
                "signal_strength": record["signal_strength"],
            }
            for record in plane_records
        ],
        "overlap_matrix": overlap_matrix,
        "audit_bridge": audit,
        "status": {
            "status_short": (
                "independent_observation_planes_ready"
                if surface_anchor_independence >= 0.95
                and source_plane_separation >= 0.95
                and variable_coupling_overlap <= 0.50
                and backfeed_risk_after_split <= 0.60
                else "independent_observation_planes_transition"
            ),
            "status_label": "独立观测面已经被拆成语言、脑编码、智能闭合和可判伪四层，但底层状态变量仍然存在明显耦合。",
        },
        "project_readout": {
            "summary": "这一轮把统一主线拆成四个观测面，先证明观测锚点可以分离，再显式暴露底层变量仍然高度耦合的事实。",
            "next_question": "下一步要把独立观测面接到强攻击测试包上，检查各平面是否会被同一反例同时打穿。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage90 Independent Observation Planes",
        "",
        f"- plane_signal_mean: {hm['plane_signal_mean']:.6f}",
        f"- surface_anchor_independence: {hm['surface_anchor_independence']:.6f}",
        f"- source_plane_separation: {hm['source_plane_separation']:.6f}",
        f"- exclusive_anchor_ratio: {hm['exclusive_anchor_ratio']:.6f}",
        f"- variable_coupling_overlap: {hm['variable_coupling_overlap']:.6f}",
        f"- backfeed_risk_after_split: {hm['backfeed_risk_after_split']:.6f}",
        f"- independent_observation_planes_score: {hm['independent_observation_planes_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_independent_observation_planes_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
