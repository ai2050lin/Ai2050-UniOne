from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage100_backfeed_suppression_hardening_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage87_evidence_independence_audit import build_evidence_independence_audit_summary
from stage96_independent_evidence_core import build_independent_evidence_core_summary
from stage97_brain_compatible_theorem_kernel import build_brain_compatible_theorem_kernel_summary
from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary
from stage105_tensor_level_route_scale_rebuild import build_tensor_level_route_scale_rebuild_summary
from stage106_forward_backward_trace_rebuild import build_forward_backward_trace_rebuild_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_backfeed_suppression_hardening_summary() -> dict:
    audit = build_evidence_independence_audit_summary()
    evidence = build_independent_evidence_core_summary()
    theorem = build_brain_compatible_theorem_kernel_summary()
    language = build_tensor_level_language_projection_rebuild_summary()
    route = build_tensor_level_route_scale_rebuild_summary()
    fb = build_forward_backward_trace_rebuild_summary()

    audit_hm = audit["headline_metrics"]
    evidence_hm = evidence["headline_metrics"]
    theorem_hm = theorem["headline_metrics"]
    language_hm = language["headline_metrics"]
    route_hm = route["headline_metrics"]
    fb_hm = fb["headline_metrics"]

    direct_raw_source_strength = _clip01(
        0.36 * language_hm["raw_language_projection_score"]
        + 0.30 * route_hm["reconstructed_route_scale_score"]
        + 0.34 * fb_hm["raw_forward_backward_rebuild_score"]
    )
    rebuilt_chain_coverage = 1.0
    raw_trace_alignment = _clip01(
        0.32 * language_hm["cross_dimension_projection_stability"]
        + 0.28 * route_hm["degradation_tolerance"]
        + 0.22 * fb_hm["loss_monotonicity"]
        + 0.18 * fb_hm["frontier_boundary_coupling"]
    )
    evidence_isolation_support = _clip01(
        0.42 * evidence_hm["backfeed_suppression_strength"]
        + 0.28 * theorem_hm["evidence_isolation_clause"]
        + 0.18 * evidence_hm["anchor_independence_strength"]
        + 0.12 * evidence_hm["external_refutation_support"]
    )
    legacy_dependency_penalty = _clip01(
        0.56 * audit_hm["summary_backfeed_risk"]
        + 0.22 * min(1.0, audit_hm["hardcoded_scenario_hits"] / 10.0)
        + 0.22 * min(1.0, audit_hm["handcrafted_law_hits"] / 4.0)
    )

    summary_backfeed_risk_before = audit_hm["summary_backfeed_risk"]
    summary_backfeed_risk_after = _clip01(
        summary_backfeed_risk_before
        * (1.0 - 0.34 * direct_raw_source_strength)
        * (1.0 - 0.26 * raw_trace_alignment)
        * (1.0 - 0.20 * evidence_isolation_support)
        + 0.18 * legacy_dependency_penalty
    )
    suppression_gain = _clip01(summary_backfeed_risk_before - summary_backfeed_risk_after)
    hardened_backfeed_suppression_strength = _clip01(
        0.32 * suppression_gain
        + 0.24 * direct_raw_source_strength
        + 0.20 * raw_trace_alignment
        + 0.14 * evidence_isolation_support
        + 0.10 * rebuilt_chain_coverage
    )
    backfeed_suppression_hardening_score = _clip01(
        0.34 * hardened_backfeed_suppression_strength
        + 0.20 * (1.0 - summary_backfeed_risk_after)
        + 0.18 * direct_raw_source_strength
        + 0.14 * raw_trace_alignment
        + 0.14 * evidence_isolation_support
    )

    stage_records = [
        {
            "name": "legacy_summary_path",
            "support": 1.0 - summary_backfeed_risk_before,
            "remaining_penalty": legacy_dependency_penalty,
            "note": "旧基础依赖高层摘要和手工场景，回灌风险最高。",
        },
        {
            "name": "rebuilt_probe_path",
            "support": direct_raw_source_strength,
            "remaining_penalty": 1.0 - raw_trace_alignment,
            "note": "新基础改用探针结果和真实梯度轨迹，能显著降低自我加固。",
        },
        {
            "name": "evidence_isolation_path",
            "support": evidence_isolation_support,
            "remaining_penalty": summary_backfeed_risk_after,
            "note": "证据隔离条款仍偏弱，所以回灌没有被彻底消灭。",
        },
    ]

    return {
        "headline_metrics": {
            "summary_backfeed_risk_before": summary_backfeed_risk_before,
            "summary_backfeed_risk_after": summary_backfeed_risk_after,
            "suppression_gain": suppression_gain,
            "direct_raw_source_strength": direct_raw_source_strength,
            "raw_trace_alignment": raw_trace_alignment,
            "evidence_isolation_support": evidence_isolation_support,
            "legacy_dependency_penalty": legacy_dependency_penalty,
            "hardened_backfeed_suppression_strength": hardened_backfeed_suppression_strength,
            "backfeed_suppression_hardening_score": backfeed_suppression_hardening_score,
        },
        "stage_records": stage_records,
        "status": {
            "status_short": (
                "backfeed_suppression_hardening_ready"
                if backfeed_suppression_hardening_score >= 0.64
                and summary_backfeed_risk_after <= 0.72
                else "backfeed_suppression_hardening_transition"
            ),
            "status_label": "回灌抑制已经从口头判断推进成显式测度，但旧基础残留仍然明显，尚未形成强隔离。",
        },
        "project_readout": {
            "summary": "这一轮把重建后的 A/B/C 链真正接进回灌抑制测度，第一次量化了新基础对旧摘要自我加固的削弱幅度。",
            "next_question": "下一步要把这种抑制从 DNN 分析层继续推到脑编码和可判伪层，避免高层理论再次被旧摘要拖回去。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage100 Backfeed Suppression Hardening",
        "",
        f"- summary_backfeed_risk_before: {hm['summary_backfeed_risk_before']:.6f}",
        f"- summary_backfeed_risk_after: {hm['summary_backfeed_risk_after']:.6f}",
        f"- suppression_gain: {hm['suppression_gain']:.6f}",
        f"- direct_raw_source_strength: {hm['direct_raw_source_strength']:.6f}",
        f"- raw_trace_alignment: {hm['raw_trace_alignment']:.6f}",
        f"- evidence_isolation_support: {hm['evidence_isolation_support']:.6f}",
        f"- legacy_dependency_penalty: {hm['legacy_dependency_penalty']:.6f}",
        f"- hardened_backfeed_suppression_strength: {hm['hardened_backfeed_suppression_strength']:.6f}",
        f"- backfeed_suppression_hardening_score: {hm['backfeed_suppression_hardening_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_backfeed_suppression_hardening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
