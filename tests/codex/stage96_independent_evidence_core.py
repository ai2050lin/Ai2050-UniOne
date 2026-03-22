from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage96_independent_evidence_core_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage87_evidence_independence_audit import build_evidence_independence_audit_summary
from stage90_independent_observation_planes import build_independent_observation_planes_summary
from stage94_cross_plane_failure_coupling_map import build_cross_plane_failure_coupling_map_summary
from stage95_external_distribution_counterexample_pack import build_external_distribution_counterexample_pack_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_independent_evidence_core_summary() -> dict:
    audit = build_evidence_independence_audit_summary()["headline_metrics"]
    planes = build_independent_observation_planes_summary()
    coupling = build_cross_plane_failure_coupling_map_summary()["headline_metrics"]
    external = build_external_distribution_counterexample_pack_summary()["headline_metrics"]

    plane_records = []
    external_receiver_focus = external["weakest_external_receiver"]
    for record in planes["plane_records"]:
        anchor_count = len(record["anchors"])
        source_count = len(record["sources"])
        anchor_strength = _clip01(record["signal_strength"] * (0.80 + 0.05 * anchor_count))
        source_diversity = _clip01(0.58 + 0.14 * source_count)
        external_support = _clip01(
            0.48 * external["external_trigger_rate"]
            + 0.24 * external["mean_strongest_path_intensity"]
            + 0.18 * external["path_alignment_rate"]
            + 0.10 * (1.0 if record["name"] == external_receiver_focus else 0.0)
        )
        plane_evidence_strength = _clip01(
            0.42 * anchor_strength
            + 0.24 * source_diversity
            + 0.20 * external_support
            + 0.14 * (1.0 - planes["headline_metrics"]["variable_coupling_overlap"])
        )
        plane_records.append(
            {
                "name": record["name"],
                "anchor_strength": anchor_strength,
                "source_diversity": source_diversity,
                "external_support": external_support,
                "plane_evidence_strength": plane_evidence_strength,
            }
        )

    anchor_independence_strength = _clip01(
        0.34 * planes["headline_metrics"]["surface_anchor_independence"]
        + 0.28 * planes["headline_metrics"]["source_plane_separation"]
        + 0.20 * planes["headline_metrics"]["exclusive_anchor_ratio"]
        + 0.18 * (1.0 - planes["headline_metrics"]["variable_coupling_overlap"])
    )
    external_refutation_support = _clip01(
        0.28 * external["external_trigger_rate"]
        + 0.24 * external["path_alignment_rate"]
        + 0.18 * external["mean_strongest_path_intensity"]
        + 0.16 * (1.0 - external["weakest_external_receiver_floor"])
        + 0.14 * external["external_distribution_counterexample_score"]
    )
    backfeed_suppression_strength = _clip01(
        0.40 * (1.0 - audit["summary_backfeed_risk"])
        + 0.30 * (1.0 - planes["headline_metrics"]["backfeed_risk_after_split"])
        + 0.18 * audit["evidence_independence_score"]
        + 0.12 * (1.0 - audit["threshold_only_test_ratio"])
    )
    cross_plane_consistency = _clip01(
        0.36 * coupling["propagation_coverage"]
        + 0.24 * coupling["cross_plane_load_mean"]
        + 0.20 * coupling["theorem_spillover_pressure"]
        + 0.20 * external["path_alignment_rate"]
    )
    independent_evidence_core_score = _clip01(
        0.28 * anchor_independence_strength
        + 0.26 * external_refutation_support
        + 0.22 * backfeed_suppression_strength
        + 0.14 * cross_plane_consistency
        + 0.10 * audit["evidence_independence_score"]
    )
    independent_ready_gap = _clip01(
        1.0
        - min(
            anchor_independence_strength,
            external_refutation_support,
            backfeed_suppression_strength,
            cross_plane_consistency,
        )
    )

    return {
        "headline_metrics": {
            "anchor_independence_strength": anchor_independence_strength,
            "external_refutation_support": external_refutation_support,
            "backfeed_suppression_strength": backfeed_suppression_strength,
            "cross_plane_consistency": cross_plane_consistency,
            "independent_ready_gap": independent_ready_gap,
            "independent_evidence_core_score": independent_evidence_core_score,
        },
        "plane_records": plane_records,
        "audit_bridge": audit,
        "external_bridge": external,
        "coupling_bridge": coupling,
        "status": {
            "status_short": (
                "independent_evidence_core_ready"
                if independent_evidence_core_score >= 0.72
                and backfeed_suppression_strength >= 0.42
                and independent_ready_gap <= 0.58
                else "independent_evidence_core_transition"
            ),
            "status_label": "独立证据主核已经开始把观测面、外部分布反例和回灌抑制压成同一评价核心，但回灌抑制仍然偏弱，所以还不是强独立证据链。",
        },
        "project_readout": {
            "summary": "这一轮开始用独立观测面、外部分布反例和回灌抑制三条线共同评估高层结论，而不再直接相信统一总分本身。",
            "next_question": "下一步要把脑编码兼容条款单独推进到定理主核，检查当前最弱链能否在不依赖摘要回灌的情况下稳定成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage96 Independent Evidence Core",
        "",
        f"- anchor_independence_strength: {hm['anchor_independence_strength']:.6f}",
        f"- external_refutation_support: {hm['external_refutation_support']:.6f}",
        f"- backfeed_suppression_strength: {hm['backfeed_suppression_strength']:.6f}",
        f"- cross_plane_consistency: {hm['cross_plane_consistency']:.6f}",
        f"- independent_ready_gap: {hm['independent_ready_gap']:.6f}",
        f"- independent_evidence_core_score: {hm['independent_evidence_core_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_independent_evidence_core_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
