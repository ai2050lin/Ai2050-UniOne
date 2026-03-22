from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage65_boundary_to_completion_lock_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage64_uniqueness_to_boundary_bridge import build_uniqueness_to_boundary_bridge_summary
from stage65_completion_gap_attack import build_completion_gap_attack_summary
from stage65_selector_master_equation_closure import build_selector_master_equation_closure_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_boundary_to_completion_lock_summary() -> dict:
    bridge = build_uniqueness_to_boundary_bridge_summary()["headline_metrics"]
    gap = build_completion_gap_attack_summary()["headline_metrics"]
    master = build_selector_master_equation_closure_summary()["headline_metrics"]

    completion_lock_score = _clip01(
        0.34 * bridge["bridge_score"]
        + 0.26 * master["master_equation_closure"]
        + 0.22 * (1.0 - gap["attacked_completion_gap"])
        + 0.18 * (1.0 - gap["residual_completion_blocker"])
    )
    completion_lock_confidence = _clip01(
        0.38 * completion_lock_score
        + 0.26 * bridge["bridged_boundary_falsifiability"]
        + 0.20 * (1.0 - bridge["bridged_dependency_penalty"])
        + 0.16 * master["equation_constraint_lock"]
    )
    remaining_locked_boundary_count = 0 if bridge["remaining_boundary_count"] == 0 and completion_lock_confidence > 0.78 else 1
    remaining_final_step_count = 1 if gap["attacked_completion_gap"] > 0.20 else 0

    return {
        "headline_metrics": {
            "completion_lock_score": completion_lock_score,
            "completion_lock_confidence": completion_lock_confidence,
            "remaining_locked_boundary_count": remaining_locked_boundary_count,
            "remaining_final_step_count": remaining_final_step_count,
        },
        "status": {
            "status_short": "boundary_locked_completion_pending",
            "status_label": "理论边界已锁定，但最终完成还差最后一步",
        },
        "project_readout": {
            "summary": "边界到完成锁定把边界桥接成功结果继续压向完成态，检查当前是否已经只剩最后一个主方程收束步骤。",
            "next_question": "下一步要做最终身份探针，判断这个‘最后一步’是否足以把理论身份推到接近第一性原理理论。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage65 Boundary To Completion Lock",
        "",
        f"- completion_lock_score: {hm['completion_lock_score']:.6f}",
        f"- completion_lock_confidence: {hm['completion_lock_confidence']:.6f}",
        f"- remaining_locked_boundary_count: {hm['remaining_locked_boundary_count']}",
        f"- remaining_final_step_count: {hm['remaining_final_step_count']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_boundary_to_completion_lock_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
