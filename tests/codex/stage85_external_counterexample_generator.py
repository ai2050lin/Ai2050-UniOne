from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage85_external_counterexample_generator_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_direct_identity_lock import build_direct_identity_lock_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage73_falsifiability_boundary_hardening import build_falsifiability_boundary_hardening_summary
from stage78_distributed_route_native_observability import build_distributed_route_native_observability_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_external_counterexample_generator_summary() -> dict:
    identity = build_direct_identity_lock_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    boundary = build_falsifiability_boundary_hardening_summary()
    route_obs = build_distributed_route_native_observability_summary()["headline_metrics"]

    boundary_bridge = boundary["boundary_bridge"]
    hb = boundary["headline_metrics"]

    samples = [
        {"name": "identity_route_split", "identity_shift": 0.78, "route_shift": 0.61, "projection_shift": 0.34, "task_shift": 0.42},
        {"name": "projection_bias_flip", "identity_shift": 0.44, "route_shift": 0.36, "projection_shift": 0.81, "task_shift": 0.40},
        {"name": "route_field_collapse", "identity_shift": 0.38, "route_shift": 0.86, "projection_shift": 0.42, "task_shift": 0.46},
        {"name": "task_identity_inversion", "identity_shift": 0.74, "route_shift": 0.48, "projection_shift": 0.40, "task_shift": 0.78},
        {"name": "multi_axis_mismatch", "identity_shift": 0.70, "route_shift": 0.68, "projection_shift": 0.66, "task_shift": 0.64},
    ]

    scenario_records = []
    trigger_count = 0
    for sample in samples:
        mismatch_energy = _clip01(
            0.30 * sample["identity_shift"]
            + 0.28 * sample["route_shift"]
            + 0.24 * sample["projection_shift"]
            + 0.18 * sample["task_shift"]
        )
        external_contact = _clip01(
            0.34 * mismatch_energy
            + 0.22 * hb["task_counterexample_activation"]
            + 0.22 * (1.0 - route_obs["field_proxy_gap"])
            + 0.22 * (1.0 - projection["projection_gap"])
        )
        support_residual = _clip01(
            0.36 * boundary_bridge["shared_state_support"]
            + 0.24 * identity["identity_lock_confidence"]
            + 0.20 * projection["language_projection_repair_score"]
            + 0.20 * route_obs["route_native_observability_score"]
        )
        refutation_strength = _clip01(
            0.50 * mismatch_energy
            + 0.28 * external_contact
            + 0.22 * max(0.0, mismatch_energy - support_residual + 0.26)
        )
        trigger_demonstrated = mismatch_energy >= 0.58 and refutation_strength >= 0.55
        if trigger_demonstrated:
            trigger_count += 1
        scenario_records.append(
            {
                "name": sample["name"],
                "generation_mode": "independent_axes",
                "derived_from_support_minus_constant": False,
                "mismatch_energy": mismatch_energy,
                "external_contact": external_contact,
                "support_residual": support_residual,
                "refutation_strength": refutation_strength,
                "trigger_demonstrated": trigger_demonstrated,
            }
        )

    strongest = max(scenario_records, key=lambda item: item["refutation_strength"])
    external_counterexample_diversity = _clip01(
        0.38 * (len(samples) / 5.0)
        + 0.34 * (sum(1 for sample in samples if sum(v > 0.6 for k, v in sample.items() if k != "name") >= 2) / len(samples))
        + 0.28 * (sum(1 for sample in samples if sample["identity_shift"] != sample["route_shift"]) / len(samples))
    )
    external_trigger_rate = trigger_count / len(scenario_records)
    shared_state_external_break_score = _clip01(
        0.30 * external_counterexample_diversity
        + 0.28 * external_trigger_rate
        + 0.24 * strongest["refutation_strength"]
        + 0.18 * strongest["external_contact"]
    )

    return {
        "headline_metrics": {
            "external_counterexample_diversity": external_counterexample_diversity,
            "external_trigger_rate": external_trigger_rate,
            "strongest_counterexample_name": strongest["name"],
            "strongest_refutation_strength": strongest["refutation_strength"],
            "shared_state_external_break_score": shared_state_external_break_score,
        },
        "scenario_records": scenario_records,
        "generation_principle": {
            "mode": "independent_axes",
            "description": "外部错配样本由身份、路由、投影、任务四个独立轴组合，不从已有支持量直接平移构造。",
        },
        "status": {
            "status_short": (
                "external_counterexample_generator_ready"
                if external_trigger_rate >= 0.20 and strongest["refutation_strength"] >= 0.55
                else "external_counterexample_generator_transition"
            ),
            "status_label": "外部反例生成块已经摆脱固定差值构造，但仍然属于内部设定样本，不是外部数据驱动反例。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage85 External Counterexample Generator",
        "",
        f"- external_counterexample_diversity: {hm['external_counterexample_diversity']:.6f}",
        f"- external_trigger_rate: {hm['external_trigger_rate']:.6f}",
        f"- strongest_counterexample_name: {hm['strongest_counterexample_name']}",
        f"- strongest_refutation_strength: {hm['strongest_refutation_strength']:.6f}",
        f"- shared_state_external_break_score: {hm['shared_state_external_break_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_external_counterexample_generator_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
