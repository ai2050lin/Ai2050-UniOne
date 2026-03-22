from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage63_first_principles_completion_possibility_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_first_principles_boundary_probe import build_first_principles_boundary_probe_summary
from stage62_transition_stability_retest import build_transition_stability_retest_summary
from stage63_global_uniqueness_constraint import build_global_uniqueness_constraint_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_first_principles_completion_possibility_summary() -> dict:
    boundary = build_first_principles_boundary_probe_summary()["headline_metrics"]
    stability = build_transition_stability_retest_summary()["headline_metrics"]
    uniqueness = build_global_uniqueness_constraint_summary()["headline_metrics"]

    theoretical_possibility_score = _clip01(
        0.30 * boundary["first_principles_readiness"]
        + 0.30 * uniqueness["mathematical_uniqueness_score"]
        + 0.20 * (1.0 - boundary["distance_to_first_principles_theory"])
        + 0.20 * boundary["boundary_falsifiability"]
    )
    completion_blocker_penalty = _clip01(
        0.32 * (1.0 - stability["stability_pass_rate"])
        + 0.24 * boundary["boundary_dependency_penalty"]
        + 0.24 * (boundary["remaining_boundary_count"] / 4.0)
        + 0.20 * (1.0 - uniqueness["mathematical_uniqueness_score"])
    )
    current_completion_readiness = _clip01(
        0.58 * theoretical_possibility_score
        + 0.16 * uniqueness["global_uniqueness_score"]
        + 0.12 * boundary["boundary_falsifiability"]
        - 0.24 * completion_blocker_penalty
    )
    remaining_completion_gap = _clip01(1.0 - current_completion_readiness)

    return {
        "headline_metrics": {
            "theoretical_possibility_score": theoretical_possibility_score,
            "completion_blocker_penalty": completion_blocker_penalty,
            "current_completion_readiness": current_completion_readiness,
            "remaining_completion_gap": remaining_completion_gap,
        },
        "status": {
            "status_short": "high_possibility_not_completed",
            "status_label": "完成第一性原理理论的可能性较高，但当前仍远未完成",
        },
        "project_readout": {
            "summary": "第一性原理完成可能性评估把全局唯一性、边界距离和稳定性惩罚并回统一判断，用来区分“理论上很可能成立”和“当前实际上已经接近完成”这两件不同的事。",
            "next_question": "下一步要优先压低 stability blocker（稳定性阻塞项）和 remaining boundaries（剩余边界），否则高可能性也难以转化成真正完成。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage63 First Principles Completion Possibility",
        "",
        f"- theoretical_possibility_score: {hm['theoretical_possibility_score']:.6f}",
        f"- completion_blocker_penalty: {hm['completion_blocker_penalty']:.6f}",
        f"- current_completion_readiness: {hm['current_completion_readiness']:.6f}",
        f"- remaining_completion_gap: {hm['remaining_completion_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_first_principles_completion_possibility_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
