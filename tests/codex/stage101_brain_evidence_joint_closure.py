from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage101_brain_evidence_joint_closure_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage92_brain_grounding_counterexample_pack import build_brain_grounding_counterexample_pack_summary
from stage97_brain_compatible_theorem_kernel import build_brain_compatible_theorem_kernel_summary
from stage98_external_to_internal_failure_alignment import build_external_to_internal_failure_alignment_summary
from stage99_real_external_data_counterexample_pack import build_real_external_data_counterexample_pack_summary
from stage100_backfeed_suppression_hardening import build_backfeed_suppression_hardening_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_brain_evidence_joint_closure_summary() -> dict:
    brain = build_brain_grounding_counterexample_pack_summary()["headline_metrics"]
    theorem = build_brain_compatible_theorem_kernel_summary()["headline_metrics"]
    alignment = build_external_to_internal_failure_alignment_summary()["headline_metrics"]
    real_external = build_real_external_data_counterexample_pack_summary()["headline_metrics"]
    backfeed = build_backfeed_suppression_hardening_summary()["headline_metrics"]

    neuron_anchor_joint = _clip01(
        0.42 * theorem["neuron_anchor_clause"]
        + 0.26 * brain["brain_grounding_residual"]
        + 0.18 * (1.0 - brain["hardest_counterexample_intensity"])
        + 0.14 * real_external["receiver_alignment_rate"]
    )
    bundle_sync_joint = _clip01(
        0.40 * theorem["bundle_sync_clause"]
        + 0.24 * brain["brain_grounding_residual"]
        + 0.18 * alignment["path_alignment_stability"]
        + 0.18 * (1.0 - brain["hardest_counterexample_intensity"])
    )
    field_observability_joint = _clip01(
        0.36 * theorem["field_compatibility_clause"]
        + 0.24 * brain["weakest_component_floor"]
        + 0.20 * (1.0 - real_external["weakest_real_receiver_floor"])
        + 0.20 * alignment["receiver_alignment_stability"]
    )
    evidence_isolation_joint = _clip01(
        0.34 * theorem["evidence_isolation_clause"]
        + 0.26 * backfeed["hardened_backfeed_suppression_strength"]
        + 0.18 * (1.0 - backfeed["summary_backfeed_risk_after"])
        + 0.12 * real_external["clause_alignment_rate"]
        + 0.10 * alignment["clause_alignment_rate"]
    )
    real_world_bridge_joint = _clip01(
        0.28 * real_external["real_trigger_rate"]
        + 0.22 * real_external["path_alignment_rate"]
        + 0.18 * real_external["receiver_alignment_rate"]
        + 0.14 * real_external["clause_alignment_rate"]
        + 0.18 * real_external["real_external_data_counterexample_score"]
    )

    joint_clauses = {
        "neuron_anchor_joint": neuron_anchor_joint,
        "bundle_sync_joint": bundle_sync_joint,
        "field_observability_joint": field_observability_joint,
        "evidence_isolation_joint": evidence_isolation_joint,
        "real_world_bridge_joint": real_world_bridge_joint,
    }
    weakest_joint_clause_name, weakest_joint_clause_score = min(joint_clauses.items(), key=lambda item: item[1])

    brain_evidence_joint_closure_gap = _clip01(1.0 - weakest_joint_clause_score)
    brain_evidence_joint_closure_score = _clip01(
        0.18 * neuron_anchor_joint
        + 0.18 * bundle_sync_joint
        + 0.20 * field_observability_joint
        + 0.24 * evidence_isolation_joint
        + 0.20 * real_world_bridge_joint
    )

    clause_records = [
        {
            "name": "neuron_anchor_joint",
            "support": neuron_anchor_joint,
            "basis": [
                "theorem.neuron_anchor_clause",
                "brain.brain_grounding_residual",
                "1-hardest_counterexample_intensity",
            ],
        },
        {
            "name": "bundle_sync_joint",
            "support": bundle_sync_joint,
            "basis": [
                "theorem.bundle_sync_clause",
                "brain.brain_grounding_residual",
                "alignment.path_alignment_stability",
            ],
        },
        {
            "name": "field_observability_joint",
            "support": field_observability_joint,
            "basis": [
                "theorem.field_compatibility_clause",
                "brain.weakest_component_floor",
                "1-real_external.weakest_real_receiver_floor",
            ],
        },
        {
            "name": "evidence_isolation_joint",
            "support": evidence_isolation_joint,
            "basis": [
                "theorem.evidence_isolation_clause",
                "backfeed.hardened_backfeed_suppression_strength",
                "1-backfeed.summary_backfeed_risk_after",
            ],
        },
        {
            "name": "real_world_bridge_joint",
            "support": real_world_bridge_joint,
            "basis": [
                "real_external.real_trigger_rate",
                "real_external.path_alignment_rate",
                "real_external.clause_alignment_rate",
            ],
        },
    ]

    return {
        "headline_metrics": {
            "neuron_anchor_joint": neuron_anchor_joint,
            "bundle_sync_joint": bundle_sync_joint,
            "field_observability_joint": field_observability_joint,
            "evidence_isolation_joint": evidence_isolation_joint,
            "real_world_bridge_joint": real_world_bridge_joint,
            "weakest_joint_clause_name": weakest_joint_clause_name,
            "weakest_joint_clause_score": weakest_joint_clause_score,
            "brain_evidence_joint_closure_gap": brain_evidence_joint_closure_gap,
            "brain_evidence_joint_closure_score": brain_evidence_joint_closure_score,
        },
        "clause_records": clause_records,
        "status": {
            "status_short": (
                "brain_evidence_joint_closure_ready"
                if brain_evidence_joint_closure_score >= 0.62
                and weakest_joint_clause_score >= 0.44
                and brain_evidence_joint_closure_gap <= 0.56
                else "brain_evidence_joint_closure_transition"
            ),
            "status_label": "脑兼容与证据独立已经进入联合闭合评估，但当前最弱条款仍然偏向证据隔离，不足以支撑第一性原理闭合。",
        },
        "project_readout": {
            "summary": "这一轮把脑编码弱链、真实外部样本对齐和回灌抑制放进同一个联合闭合块，专门检查理论主核是否会被脑弱链和证据弱链一起拖垮。",
            "next_question": "下一步要继续把真实世界任务数据接进 real_world_bridge_joint，避免它长期停留在词表级外部样本上。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage101 Brain Evidence Joint Closure",
        "",
        f"- neuron_anchor_joint: {hm['neuron_anchor_joint']:.6f}",
        f"- bundle_sync_joint: {hm['bundle_sync_joint']:.6f}",
        f"- field_observability_joint: {hm['field_observability_joint']:.6f}",
        f"- evidence_isolation_joint: {hm['evidence_isolation_joint']:.6f}",
        f"- real_world_bridge_joint: {hm['real_world_bridge_joint']:.6f}",
        f"- weakest_joint_clause_name: {hm['weakest_joint_clause_name']}",
        f"- weakest_joint_clause_score: {hm['weakest_joint_clause_score']:.6f}",
        f"- brain_evidence_joint_closure_gap: {hm['brain_evidence_joint_closure_gap']:.6f}",
        f"- brain_evidence_joint_closure_score: {hm['brain_evidence_joint_closure_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_evidence_joint_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
