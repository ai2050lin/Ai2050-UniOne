from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage88_external_counterexample_expansion_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage73_falsifiability_boundary_hardening import build_falsifiability_boundary_hardening_summary
from stage78_distributed_route_native_observability import build_distributed_route_native_observability_summary
from stage85_external_counterexample_generator import build_external_counterexample_generator_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_external_counterexample_expansion_summary() -> dict:
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    boundary = build_falsifiability_boundary_hardening_summary()
    route_obs = build_distributed_route_native_observability_summary()["headline_metrics"]
    external = build_external_counterexample_generator_summary()

    boundary_bridge = boundary["boundary_bridge"]
    hb = boundary["headline_metrics"]
    generator_hm = external["headline_metrics"]

    expanded_samples = [
        {"name": "identity_route_projection_triplet", "family": "triple_axis", "identity_shift": 0.82, "route_shift": 0.76, "projection_shift": 0.72, "task_shift": 0.44, "stress_bonus": 0.08},
        {"name": "task_projection_reversal", "family": "task_projection", "identity_shift": 0.58, "route_shift": 0.40, "projection_shift": 0.84, "task_shift": 0.78, "stress_bonus": 0.07},
        {"name": "route_field_permutation", "family": "order_shuffle", "identity_shift": 0.48, "route_shift": 0.88, "projection_shift": 0.56, "task_shift": 0.52, "stress_bonus": 0.09},
        {"name": "boundary_load_overflow", "family": "boundary_overflow", "identity_shift": 0.62, "route_shift": 0.78, "projection_shift": 0.66, "task_shift": 0.82, "stress_bonus": 0.11},
        {"name": "shared_state_axis_swap", "family": "axis_swap", "identity_shift": 0.76, "route_shift": 0.74, "projection_shift": 0.38, "task_shift": 0.58, "stress_bonus": 0.06},
        {"name": "projection_route_lock_conflict", "family": "route_projection", "identity_shift": 0.54, "route_shift": 0.82, "projection_shift": 0.80, "task_shift": 0.46, "stress_bonus": 0.08},
        {"name": "multi_axis_extreme", "family": "extreme", "identity_shift": 0.88, "route_shift": 0.86, "projection_shift": 0.84, "task_shift": 0.80, "stress_bonus": 0.12},
        {"name": "task_identity_projection_quartet", "family": "quartet", "identity_shift": 0.80, "route_shift": 0.62, "projection_shift": 0.78, "task_shift": 0.76, "stress_bonus": 0.09},
    ]

    scenario_records = []
    trigger_count = 0
    family_counts: dict[str, int] = {}
    family_trigger_counts: dict[str, int] = {}
    for sample in expanded_samples:
        family_counts[sample["family"]] = family_counts.get(sample["family"], 0) + 1
        mismatch_energy = _clip01(
            0.28 * sample["identity_shift"]
            + 0.28 * sample["route_shift"]
            + 0.24 * sample["projection_shift"]
            + 0.20 * sample["task_shift"]
        )
        external_contact = _clip01(
            0.30 * mismatch_energy
            + 0.18 * sample["stress_bonus"]
            + 0.18 * hb["task_counterexample_activation"]
            + 0.18 * (1.0 - route_obs["field_proxy_gap"])
            + 0.16 * (1.0 - projection["projection_gap"])
        )
        support_residual = _clip01(
            0.32 * boundary_bridge["shared_state_support"]
            + 0.22 * projection["language_projection_repair_score"]
            + 0.22 * route_obs["route_native_observability_score"]
            + 0.12 * hb["shared_state_rejection_power"]
            + 0.12 * generator_hm["shared_state_external_break_score"]
        )
        refutation_strength = _clip01(
            0.44 * mismatch_energy
            + 0.24 * external_contact
            + 0.20 * sample["stress_bonus"]
            + 0.12 * max(0.0, mismatch_energy - support_residual + 0.18)
        )
        trigger_demonstrated = mismatch_energy >= 0.66 and refutation_strength >= 0.46
        if trigger_demonstrated:
            trigger_count += 1
            family_trigger_counts[sample["family"]] = family_trigger_counts.get(sample["family"], 0) + 1
        scenario_records.append(
            {
                "name": sample["name"],
                "family": sample["family"],
                "generation_mode": "expanded_independent_axes",
                "mismatch_energy": mismatch_energy,
                "external_contact": external_contact,
                "support_residual": support_residual,
                "refutation_strength": refutation_strength,
                "trigger_demonstrated": trigger_demonstrated,
            }
        )

    strongest = max(scenario_records, key=lambda item: item["refutation_strength"])
    expanded_trigger_rate = trigger_count / len(scenario_records)
    family_coverage = _clip01(len(family_counts) / len(expanded_samples))
    triggered_family_coverage = len(family_trigger_counts) / len(family_counts)
    average_refutation_strength = sum(item["refutation_strength"] for item in scenario_records) / len(scenario_records)
    external_counterexample_expansion_score = _clip01(
        0.24 * family_coverage
        + 0.24 * triggered_family_coverage
        + 0.22 * expanded_trigger_rate
        + 0.18 * strongest["refutation_strength"]
        + 0.12 * average_refutation_strength
    )

    return {
        "headline_metrics": {
            "family_coverage": family_coverage,
            "triggered_family_coverage": triggered_family_coverage,
            "expanded_trigger_rate": expanded_trigger_rate,
            "strongest_counterexample_name": strongest["name"],
            "strongest_refutation_strength": strongest["refutation_strength"],
            "average_refutation_strength": average_refutation_strength,
            "external_counterexample_expansion_score": external_counterexample_expansion_score,
        },
        "scenario_records": scenario_records,
        "status": {
            "status_short": (
                "external_counterexample_expansion_ready"
                if expanded_trigger_rate >= 0.40 and strongest["refutation_strength"] >= 0.58
                else "external_counterexample_expansion_transition"
            ),
            "status_label": "外部反例扩展块已经能扩大样本族并提升触发率，但仍然属于内部构造反例，而不是外部真实分布采样。",
        },
        "project_readout": {
            "summary": "这一轮把外部反例从 5 个独立样本扩展到 8 个跨家族样本，开始覆盖三轴、四轴、顺序打乱和边界过载等更强错配类型。",
            "next_question": "下一步要把扩展反例直接回灌到可判伪主核，检查最坏反例强度是否继续抬升。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage88 External Counterexample Expansion",
        "",
        f"- family_coverage: {hm['family_coverage']:.6f}",
        f"- triggered_family_coverage: {hm['triggered_family_coverage']:.6f}",
        f"- expanded_trigger_rate: {hm['expanded_trigger_rate']:.6f}",
        f"- strongest_counterexample_name: {hm['strongest_counterexample_name']}",
        f"- strongest_refutation_strength: {hm['strongest_refutation_strength']:.6f}",
        f"- average_refutation_strength: {hm['average_refutation_strength']:.6f}",
        f"- external_counterexample_expansion_score: {hm['external_counterexample_expansion_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_external_counterexample_expansion_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
