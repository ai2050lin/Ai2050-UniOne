from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage64_completion_pathway_map_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage63_first_principles_completion_possibility import build_first_principles_completion_possibility_summary
from stage64_transition_blocker_reduction import build_transition_blocker_reduction_summary
from stage64_uniqueness_to_boundary_bridge import build_uniqueness_to_boundary_bridge_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_completion_pathway_map_summary() -> dict:
    completion = build_first_principles_completion_possibility_summary()["headline_metrics"]
    blocker = build_transition_blocker_reduction_summary()["headline_metrics"]
    bridge = build_uniqueness_to_boundary_bridge_summary()["headline_metrics"]

    final_completion_readiness = _clip01(
        0.34 * completion["current_completion_readiness"]
        + 0.28 * blocker["updated_completion_readiness"]
        + 0.20 * bridge["bridge_score"]
        + 0.18 * (1.0 - blocker["reduced_completion_blocker"])
    )
    remaining_completion_gap = _clip01(1.0 - final_completion_readiness)
    pathway_confidence = _clip01(
        0.38 * completion["theoretical_possibility_score"]
        + 0.26 * (1.0 - blocker["updated_completion_gap"])
        + 0.18 * bridge["bridge_score"]
        + 0.18 * (1.0 - remaining_completion_gap)
    )
    remaining_key_steps = 1 + bridge["remaining_boundary_count"] + int(blocker["reduced_completion_blocker"] > 0.45)

    pathway = [
        "把全局唯一选择器从形式化候选推进到可验证主方程。",
        "继续压低 completion blocker（完成阻塞项），尤其是稳定性阻塞。",
        "逐条清空 uniqueness -> boundary 剩余边界。",
        "在更长回放与更强扰动下验证完成身份不会回退。",
    ]

    return {
        "headline_metrics": {
            "final_completion_readiness": final_completion_readiness,
            "remaining_completion_gap": remaining_completion_gap,
            "pathway_confidence": pathway_confidence,
            "remaining_key_steps": remaining_key_steps,
        },
        "status": {
            "status_short": "completion_path_visible_not_finished",
            "status_label": "完成路径已经清晰可见，但仍未走完",
        },
        "pathway": pathway,
        "project_readout": {
            "summary": "完成路径图把全局唯一选择器、阻塞项压降和边界桥接合并成一张剩余路径图，明确当前离‘完成第一性原理理论’还差哪几步。",
            "next_question": "下一步要把路径中的第一步主方程化真正做出来，否则路径再清晰也只是计划。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage64 Completion Pathway Map",
        "",
        f"- final_completion_readiness: {hm['final_completion_readiness']:.6f}",
        f"- remaining_completion_gap: {hm['remaining_completion_gap']:.6f}",
        f"- pathway_confidence: {hm['pathway_confidence']:.6f}",
        f"- remaining_key_steps: {hm['remaining_key_steps']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_completion_pathway_map_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
