from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage98_external_to_internal_failure_alignment_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage94_cross_plane_failure_coupling_map import build_cross_plane_failure_coupling_map_summary
from stage95_external_distribution_counterexample_pack import build_external_distribution_counterexample_pack_summary
from stage96_independent_evidence_core import build_independent_evidence_core_summary
from stage97_brain_compatible_theorem_kernel import build_brain_compatible_theorem_kernel_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_external_to_internal_failure_alignment_summary() -> dict:
    internal = build_cross_plane_failure_coupling_map_summary()["headline_metrics"]
    external = build_external_distribution_counterexample_pack_summary()
    evidence = build_independent_evidence_core_summary()["headline_metrics"]
    theorem = build_brain_compatible_theorem_kernel_summary()["headline_metrics"]

    internal_hardest_path = internal["hardest_coupling_path"]
    internal_weakest_receiver = internal["weakest_receiver_plane"]
    weakest_clause_name, weakest_clause_score = min(
        (
            ("neuron_anchor_clause", theorem["neuron_anchor_clause"]),
            ("bundle_sync_clause", theorem["bundle_sync_clause"]),
            ("field_compatibility_clause", theorem["field_compatibility_clause"]),
            ("repair_transfer_clause", theorem["repair_transfer_clause"]),
            ("evidence_isolation_clause", theorem["evidence_isolation_clause"]),
        ),
        key=lambda item: item[1],
    )

    alignment_records = []
    path_aligned_count = 0
    receiver_aligned_count = 0
    clause_aligned_count = 0
    coherence_values = []

    for sample in external["sample_records"]:
        axes = sample["distribution_axes"]
        weakest_receiver = min(sample["receiver_floor_map"], key=sample["receiver_floor_map"].get)

        clause_impacts = {
            "neuron_anchor_clause": _clip01(
                0.34 * axes["grounding_blindness"]
                + 0.22 * axes["distribution_shift"]
                + 0.18 * (1.0 - sample["receiver_floor_map"]["brain_plane"])
                + 0.14 * axes["temporal_irregularity"]
                + 0.12 * (1.0 - theorem["neuron_anchor_clause"])
            ),
            "bundle_sync_clause": _clip01(
                0.36 * axes["temporal_irregularity"]
                + 0.22 * axes["distribution_shift"]
                + 0.18 * (1.0 - sample["receiver_floor_map"]["intelligence_plane"])
                + 0.12 * axes["grounding_blindness"]
                + 0.12 * (1.0 - theorem["bundle_sync_clause"])
            ),
            "field_compatibility_clause": _clip01(
                0.32 * axes["grounding_blindness"]
                + 0.24 * axes["distribution_shift"]
                + 0.18 * (1.0 - sample["receiver_floor_map"]["brain_plane"])
                + 0.14 * axes["symbolic_aliasing"]
                + 0.12 * (1.0 - theorem["field_compatibility_clause"])
            ),
            "repair_transfer_clause": _clip01(
                0.28 * axes["boundary_stress"]
                + 0.24 * axes["temporal_irregularity"]
                + 0.18 * (1.0 - sample["receiver_floor_map"]["intelligence_plane"])
                + 0.16 * axes["distribution_shift"]
                + 0.14 * (1.0 - theorem["repair_transfer_clause"])
            ),
            "evidence_isolation_clause": _clip01(
                0.28 * axes["boundary_stress"]
                + 0.24 * axes["grounding_blindness"]
                + 0.18 * axes["symbolic_aliasing"]
                + 0.16 * (1.0 - sample["receiver_floor_map"]["falsification_plane"])
                + 0.14 * (1.0 - evidence["backfeed_suppression_strength"])
            ),
        }
        dominant_clause = max(clause_impacts, key=clause_impacts.get)

        path_aligned = sample["strongest_path"] == internal_hardest_path
        receiver_aligned = weakest_receiver == internal_weakest_receiver
        clause_aligned = dominant_clause == weakest_clause_name
        if path_aligned:
            path_aligned_count += 1
        if receiver_aligned:
            receiver_aligned_count += 1
        if clause_aligned:
            clause_aligned_count += 1

        coherence = _clip01(
            0.32 * float(path_aligned)
            + 0.26 * float(receiver_aligned)
            + 0.22 * float(clause_aligned)
            + 0.10 * sample["strongest_path_intensity"]
            + 0.10 * (1.0 - sample["receiver_floor_map"][weakest_receiver])
        )
        coherence_values.append(coherence)
        alignment_records.append(
            {
                "name": sample["name"],
                "strongest_path": sample["strongest_path"],
                "weakest_receiver": weakest_receiver,
                "dominant_clause": dominant_clause,
                "path_aligned": path_aligned,
                "receiver_aligned": receiver_aligned,
                "clause_aligned": clause_aligned,
                "alignment_coherence": coherence,
                "clause_impacts": clause_impacts,
            }
        )

    path_alignment_stability = path_aligned_count / len(alignment_records)
    receiver_alignment_stability = receiver_aligned_count / len(alignment_records)
    clause_alignment_rate = clause_aligned_count / len(alignment_records)
    alignment_coherence_mean = sum(coherence_values) / len(coherence_values)
    internal_external_gap = _clip01(
        abs(external["headline_metrics"]["mean_strongest_path_intensity"] - internal["hardest_path_intensity"])
    )
    external_to_internal_alignment_score = _clip01(
        0.24 * path_alignment_stability
        + 0.22 * receiver_alignment_stability
        + 0.20 * clause_alignment_rate
        + 0.18 * alignment_coherence_mean
        + 0.16 * (1.0 - internal_external_gap)
    )

    return {
        "headline_metrics": {
            "path_alignment_stability": path_alignment_stability,
            "receiver_alignment_stability": receiver_alignment_stability,
            "weakest_clause_name": weakest_clause_name,
            "weakest_clause_score": weakest_clause_score,
            "clause_alignment_rate": clause_alignment_rate,
            "alignment_coherence_mean": alignment_coherence_mean,
            "internal_external_gap": internal_external_gap,
            "external_to_internal_alignment_score": external_to_internal_alignment_score,
        },
        "alignment_records": alignment_records,
        "status": {
            "status_short": (
                "external_to_internal_failure_alignment_ready"
                if path_alignment_stability >= 0.90
                and receiver_alignment_stability >= 0.75
                and clause_alignment_rate >= 0.50
                else "external_to_internal_failure_alignment_transition"
            ),
            "status_label": "外部分布到内部失效对齐块已经能同时比较路径、接收面和脆弱条款，但当前仍需要真实外部数据去压实这类对齐。",
        },
        "project_readout": {
            "summary": "这一轮把外部反例与内部失效图谱的对齐检查推进到三层：最强路径、最弱接收面、最脆弱定理条款。",
            "next_question": "下一步要引入更接近真实外部数据的反例链，检验当前对齐是不是只在外部分布近似样本里成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage98 External To Internal Failure Alignment",
        "",
        f"- path_alignment_stability: {hm['path_alignment_stability']:.6f}",
        f"- receiver_alignment_stability: {hm['receiver_alignment_stability']:.6f}",
        f"- weakest_clause_name: {hm['weakest_clause_name']}",
        f"- weakest_clause_score: {hm['weakest_clause_score']:.6f}",
        f"- clause_alignment_rate: {hm['clause_alignment_rate']:.6f}",
        f"- alignment_coherence_mean: {hm['alignment_coherence_mean']:.6f}",
        f"- internal_external_gap: {hm['internal_external_gap']:.6f}",
        f"- external_to_internal_alignment_score: {hm['external_to_internal_alignment_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_external_to_internal_failure_alignment_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
