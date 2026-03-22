from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage97_brain_compatible_theorem_kernel_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage92_brain_grounding_counterexample_pack import build_brain_grounding_counterexample_pack_summary
from stage93_law_to_theorem_bridge import build_law_to_theorem_bridge_summary
from stage95_external_distribution_counterexample_pack import build_external_distribution_counterexample_pack_summary
from stage96_independent_evidence_core import build_independent_evidence_core_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_brain_compatible_theorem_kernel_summary() -> dict:
    brain = build_brain_grounding_counterexample_pack_summary()
    theorem = build_law_to_theorem_bridge_summary()
    external = build_external_distribution_counterexample_pack_summary()
    evidence = build_independent_evidence_core_summary()

    brain_hm = brain["headline_metrics"]
    theorem_hm = theorem["headline_metrics"]
    external_hm = external["headline_metrics"]
    evidence_hm = evidence["headline_metrics"]

    component_means = {}
    for component_name in brain["scenario_records"][0]["component_after"]:
        component_means[component_name] = sum(
            record["component_after"][component_name] for record in brain["scenario_records"]
        ) / len(brain["scenario_records"])

    neuron_anchor_clause = _clip01(
        0.46 * component_means["neuron_anchor"]
        + 0.26 * brain_hm["brain_grounding_residual"]
        + 0.16 * theorem_hm["brain_compatibility_clause"]
        + 0.12 * evidence_hm["anchor_independence_strength"]
    )
    bundle_sync_clause = _clip01(
        0.42 * component_means["bundle_sync"]
        + 0.24 * (1.0 - brain_hm["hardest_counterexample_intensity"])
        + 0.18 * external_hm["path_alignment_rate"]
        + 0.16 * evidence_hm["cross_plane_consistency"]
    )
    field_compatibility_clause = _clip01(
        0.34 * component_means["distributed_field"]
        + 0.34 * component_means["field_observability"]
        + 0.20 * theorem_hm["brain_compatibility_clause"]
        + 0.12 * (1.0 - external_hm["weakest_external_receiver_floor"])
    )
    repair_transfer_clause = _clip01(
        0.36 * component_means["repair_grounding"]
        + 0.22 * theorem_hm["boundary_clause_strength"]
        + 0.22 * evidence_hm["cross_plane_consistency"]
        + 0.20 * external_hm["mean_strongest_path_intensity"]
    )
    evidence_isolation_clause = _clip01(
        0.48 * evidence_hm["backfeed_suppression_strength"]
        + 0.24 * evidence_hm["external_refutation_support"]
        + 0.16 * evidence_hm["anchor_independence_strength"]
        + 0.12 * (1.0 - evidence_hm["independent_ready_gap"])
    )

    theorem_viability_gap = _clip01(
        1.0
        - min(
            neuron_anchor_clause,
            bundle_sync_clause,
            field_compatibility_clause,
            repair_transfer_clause,
            evidence_isolation_clause,
        )
    )
    brain_compatible_theorem_kernel_score = _clip01(
        0.20 * neuron_anchor_clause
        + 0.20 * bundle_sync_clause
        + 0.22 * field_compatibility_clause
        + 0.18 * repair_transfer_clause
        + 0.12 * evidence_isolation_clause
        + 0.08 * (1.0 - theorem_viability_gap)
    )

    clause_records = [
        {
            "name": "neuron_anchor_clause",
            "support": neuron_anchor_clause,
            "basis": [
                "component_mean(neuron_anchor)",
                "brain_grounding_residual",
                "brain_compatibility_clause",
            ],
        },
        {
            "name": "bundle_sync_clause",
            "support": bundle_sync_clause,
            "basis": [
                "component_mean(bundle_sync)",
                "hardest_counterexample_intensity",
                "path_alignment_rate",
            ],
        },
        {
            "name": "field_compatibility_clause",
            "support": field_compatibility_clause,
            "basis": [
                "component_mean(distributed_field)",
                "component_mean(field_observability)",
                "brain_compatibility_clause",
            ],
        },
        {
            "name": "repair_transfer_clause",
            "support": repair_transfer_clause,
            "basis": [
                "component_mean(repair_grounding)",
                "boundary_clause_strength",
                "cross_plane_consistency",
            ],
        },
        {
            "name": "evidence_isolation_clause",
            "support": evidence_isolation_clause,
            "basis": [
                "backfeed_suppression_strength",
                "external_refutation_support",
                "anchor_independence_strength",
            ],
        },
    ]

    return {
        "headline_metrics": {
            "neuron_anchor_clause": neuron_anchor_clause,
            "bundle_sync_clause": bundle_sync_clause,
            "field_compatibility_clause": field_compatibility_clause,
            "repair_transfer_clause": repair_transfer_clause,
            "evidence_isolation_clause": evidence_isolation_clause,
            "theorem_viability_gap": theorem_viability_gap,
            "brain_compatible_theorem_kernel_score": brain_compatible_theorem_kernel_score,
        },
        "component_means": component_means,
        "clause_records": clause_records,
        "status": {
            "status_short": (
                "brain_compatible_theorem_kernel_ready"
                if brain_compatible_theorem_kernel_score >= 0.68
                and evidence_isolation_clause >= 0.44
                and theorem_viability_gap <= 0.56
                else "brain_compatible_theorem_kernel_transition"
            ),
            "status_label": "脑兼容定理主核已经把脑编码弱链拆成定理条款，但证据隔离条款仍然偏弱，所以当前仍是过渡态。",
        },
        "project_readout": {
            "summary": "这一轮把脑兼容问题从单个弱分数推进成了 5 个定理条款：局部锚点、束流同步、场兼容、修复传递、证据隔离。",
            "next_question": "下一步要检查外部分布反例触发的最强路径，是否会稳定打穿同一条脑兼容条款，而不是只在内部样本上成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage97 Brain Compatible Theorem Kernel",
        "",
        f"- neuron_anchor_clause: {hm['neuron_anchor_clause']:.6f}",
        f"- bundle_sync_clause: {hm['bundle_sync_clause']:.6f}",
        f"- field_compatibility_clause: {hm['field_compatibility_clause']:.6f}",
        f"- repair_transfer_clause: {hm['repair_transfer_clause']:.6f}",
        f"- evidence_isolation_clause: {hm['evidence_isolation_clause']:.6f}",
        f"- theorem_viability_gap: {hm['theorem_viability_gap']:.6f}",
        f"- brain_compatible_theorem_kernel_score: {hm['brain_compatible_theorem_kernel_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_compatible_theorem_kernel_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
