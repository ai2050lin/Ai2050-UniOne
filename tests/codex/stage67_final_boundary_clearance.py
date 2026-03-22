from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage67_final_boundary_clearance_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_first_principles_boundary_probe import build_first_principles_boundary_probe_summary
from stage65_boundary_to_completion_lock import build_boundary_to_completion_lock_summary
from stage66_first_principles_convergence_assessment import build_first_principles_convergence_assessment_summary
from stage67_uniqueness_gap_reduction import build_uniqueness_gap_reduction_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_summary(relpath: str, builder) -> dict:
    path = ROOT / relpath
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return builder()


def build_final_boundary_clearance_summary() -> dict:
    boundary = build_first_principles_boundary_probe_summary()["headline_metrics"]
    lock = build_boundary_to_completion_lock_summary()["headline_metrics"]
    convergence = _load_summary(
        "tests/codex_temp/stage66_first_principles_convergence_assessment_20260322/summary.json",
        build_first_principles_convergence_assessment_summary,
    )["headline_metrics"]
    proof = _load_summary(
        "tests/codex_temp/stage67_uniqueness_gap_reduction_20260322/summary.json",
        build_uniqueness_gap_reduction_summary,
    )["headline_metrics"]

    final_boundary_clearance = _clip01(
        0.30 * convergence["convergence_closure"]
        + 0.26 * convergence["convergence_falsifiability"]
        + 0.20 * (1.0 - convergence["convergence_dependency_penalty"])
        + 0.24 * (1.0 - proof["reduced_proof_gap"])
    )
    boundary_lock_confidence = _clip01(
        0.32 * final_boundary_clearance
        + 0.28 * lock["completion_lock_confidence"]
        + 0.20 * (1.0 - boundary["distance_to_first_principles_theory"])
        + 0.20 * (1.0 - proof["reduced_proof_gap"])
    )
    remaining_boundary_count = 0 if final_boundary_clearance > 0.78 and boundary_lock_confidence > 0.76 else 1

    return {
        "headline_metrics": {
            "final_boundary_clearance": final_boundary_clearance,
            "boundary_lock_confidence": boundary_lock_confidence,
            "remaining_boundary_count": remaining_boundary_count,
        },
        "status": {
            "status_short": "final_boundary_nearly_cleared" if remaining_boundary_count == 0 else "final_boundary_not_cleared",
            "status_label": "最后边界已经被逼到接近清零的位置，但是否真正清零仍取决于证明缺口和锁定信心是否同时跨线",
        },
        "project_readout": {
            "summary": "这一轮把理论边界、完成锁定、收束判断和证明缺口重新合流，专门判断最后那一道边界是否真的已经被清到 0。",
            "next_question": "下一步要把最后边界清零结果与身份切换探针直接并回，避免出现边界看似清零、身份却没有变化的伪进展。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage67 Final Boundary Clearance",
        "",
        f"- final_boundary_clearance: {hm['final_boundary_clearance']:.6f}",
        f"- boundary_lock_confidence: {hm['boundary_lock_confidence']:.6f}",
        f"- remaining_boundary_count: {hm['remaining_boundary_count']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_final_boundary_clearance_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
