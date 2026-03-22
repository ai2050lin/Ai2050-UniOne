from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage62_first_principles_boundary_probe_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage61_theory_identity_retest import build_theory_identity_retest_summary
from stage62_low_dependency_band_stress import build_low_dependency_band_stress_summary
from stage62_transition_stability_retest import build_transition_stability_retest_summary
from stage62_uniqueness_hardening import build_uniqueness_hardening_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_first_principles_boundary_probe_summary() -> dict:
    retest = build_theory_identity_retest_summary()["headline_metrics"]
    stability = build_transition_stability_retest_summary()["headline_metrics"]
    band_summary = build_low_dependency_band_stress_summary()
    band = band_summary["headline_metrics"]
    uniq = build_uniqueness_hardening_summary()["headline_metrics"]
    safe_points = [point for point in band_summary["stressed_points"] if point["stressed_safe"]]
    safe_penalty_floor = min(point["stressed_penalty"] for point in safe_points)

    boundary_closure = _clip01(
        0.38 * retest["retest_closure"]
        + 0.22 * stability["avg_closure"]
        + 0.20 * uniq["hardened_uniqueness_score"]
        + 0.20 * band["band_resilience_score"]
    )
    boundary_falsifiability = _clip01(
        0.40 * retest["retest_falsifiability"]
        + 0.28 * stability["avg_falsifiability"]
        + 0.12 * stability["transition_stability_score"]
        + 0.20 * uniq["cross_task_lock_score"]
    )
    boundary_dependency_penalty = _clip01(
        0.44 * retest["retest_dependency_penalty"]
        + 0.28 * stability["avg_dependency_penalty"]
        + 0.18 * safe_penalty_floor
        + 0.10 * (1.0 - band["band_resilience_score"])
    )
    first_principles_readiness = _clip01(
        0.30 * boundary_closure
        + 0.28 * boundary_falsifiability
        + 0.20 * (1.0 - boundary_dependency_penalty)
        + 0.22 * uniq["hardened_uniqueness_score"]
    )

    closure_gap = max(0.0, 0.68 - boundary_closure)
    falsifiability_gap = max(0.0, 0.78 - boundary_falsifiability)
    dependency_gap = max(0.0, boundary_dependency_penalty - 0.50)
    uniqueness_gap = uniq["residual_uniqueness_gap"]
    distance_to_first_principles_theory = _clip01(
        0.30 * closure_gap
        + 0.22 * falsifiability_gap
        + 0.28 * dependency_gap
        + 0.20 * uniqueness_gap
    )
    remaining_boundary_count = sum(
        1
        for value in (closure_gap > 0.0, falsifiability_gap > 0.0, dependency_gap > 0.0, uniqueness_gap > 0.12)
        if value
    )

    return {
        "headline_metrics": {
            "boundary_closure": boundary_closure,
            "boundary_falsifiability": boundary_falsifiability,
            "boundary_dependency_penalty": boundary_dependency_penalty,
            "first_principles_readiness": first_principles_readiness,
            "distance_to_first_principles_theory": distance_to_first_principles_theory,
            "remaining_boundary_count": remaining_boundary_count,
        },
        "status": {
            "status_short": "phenomenological_transition",
            "status_label": "仍属唯象模型，但已进入第一性原理过渡区",
        },
        "project_readout": {
            "summary": "第一性原理边界探针把稳定性、低依赖带和唯一化加固一起并入，专门测量当前体系离真正第一性原理理论还剩下多远。",
            "next_question": "下一步要针对 remaining boundaries 逐个做定向突破，而不是再平均抬高所有分数。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage62 First Principles Boundary Probe",
        "",
        f"- boundary_closure: {hm['boundary_closure']:.6f}",
        f"- boundary_falsifiability: {hm['boundary_falsifiability']:.6f}",
        f"- boundary_dependency_penalty: {hm['boundary_dependency_penalty']:.6f}",
        f"- first_principles_readiness: {hm['first_principles_readiness']:.6f}",
        f"- distance_to_first_principles_theory: {hm['distance_to_first_principles_theory']:.6f}",
        f"- remaining_boundary_count: {hm['remaining_boundary_count']}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_first_principles_boundary_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
