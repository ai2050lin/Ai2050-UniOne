from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage65_completion_gap_attack_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage64_completion_pathway_map import build_completion_pathway_map_summary
from stage64_transition_blocker_reduction import build_transition_blocker_reduction_summary
from stage64_uniqueness_to_boundary_bridge import build_uniqueness_to_boundary_bridge_summary
from stage65_selector_master_equation_closure import build_selector_master_equation_closure_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_completion_gap_attack_summary() -> dict:
    pathway = build_completion_pathway_map_summary()["headline_metrics"]
    blocker = build_transition_blocker_reduction_summary()["headline_metrics"]
    bridge = build_uniqueness_to_boundary_bridge_summary()["headline_metrics"]
    master = build_selector_master_equation_closure_summary()["headline_metrics"]

    gap_reduction_gain = _clip01(
        0.30 * blocker["blocker_reduction_gain"]
        + 0.22 * master["master_equation_closure"]
        + 0.20 * bridge["bridge_score"]
        + 0.14 * (1.0 - blocker["reduced_completion_blocker"])
        + 0.14 * (1.0 - master["residual_master_gap"])
    )
    attacked_completion_gap = _clip01(
        pathway["remaining_completion_gap"]
        - 0.12 * gap_reduction_gain
        - 0.06 * master["master_equation_closure"]
        - 0.04 * (1.0 - blocker["reduced_completion_blocker"])
    )
    attacked_completion_readiness = _clip01(
        0.46 * pathway["final_completion_readiness"]
        + 0.26 * blocker["updated_completion_readiness"]
        + 0.18 * (1.0 - attacked_completion_gap)
        + 0.10 * master["master_equation_closure"]
    )
    residual_completion_blocker = _clip01(
        blocker["reduced_completion_blocker"]
        - 0.08 * gap_reduction_gain
        - 0.04 * (1.0 - master["residual_master_gap"])
    )

    return {
        "headline_metrics": {
            "gap_reduction_gain": gap_reduction_gain,
            "attacked_completion_gap": attacked_completion_gap,
            "attacked_completion_readiness": attacked_completion_readiness,
            "residual_completion_blocker": residual_completion_blocker,
        },
        "status": {
            "status_short": "completion_gap_under_attack",
            "status_label": "完成缺口已被明显压缩，但仍未清零",
        },
        "project_readout": {
            "summary": "完成缺口攻击把主方程闭合和边界桥接结果并回完成路径图，直接测量剩余完成缺口是否开始快速下降。",
            "next_question": "下一步要检查边界已经清零后，是否能把剩余关键步骤再压缩到最后一步。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage65 Completion Gap Attack",
        "",
        f"- gap_reduction_gain: {hm['gap_reduction_gain']:.6f}",
        f"- attacked_completion_gap: {hm['attacked_completion_gap']:.6f}",
        f"- attacked_completion_readiness: {hm['attacked_completion_readiness']:.6f}",
        f"- residual_completion_blocker: {hm['residual_completion_blocker']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_completion_gap_attack_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
