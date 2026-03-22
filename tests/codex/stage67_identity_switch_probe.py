from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage67_identity_switch_probe_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage66_first_principles_convergence_assessment import build_first_principles_convergence_assessment_summary
from stage67_context_fiber_primitive_repair import build_context_fiber_primitive_repair_summary
from stage67_final_boundary_clearance import build_final_boundary_clearance_summary
from stage67_uniqueness_gap_reduction import build_uniqueness_gap_reduction_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_summary(relpath: str, builder) -> dict:
    path = ROOT / relpath
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return builder()


def build_identity_switch_probe_summary() -> dict:
    convergence = _load_summary(
        "tests/codex_temp/stage66_first_principles_convergence_assessment_20260322/summary.json",
        build_first_principles_convergence_assessment_summary,
    )["headline_metrics"]
    repair = _load_summary(
        "tests/codex_temp/stage67_context_fiber_primitive_repair_20260322/summary.json",
        build_context_fiber_primitive_repair_summary,
    )["headline_metrics"]
    boundary = _load_summary(
        "tests/codex_temp/stage67_final_boundary_clearance_20260322/summary.json",
        build_final_boundary_clearance_summary,
    )["headline_metrics"]
    proof = _load_summary(
        "tests/codex_temp/stage67_uniqueness_gap_reduction_20260322/summary.json",
        build_uniqueness_gap_reduction_summary,
    )["headline_metrics"]

    switched_closure = _clip01(
        0.42 * convergence["convergence_closure"]
        + 0.26 * repair["repaired_primitive_closure"]
        + 0.16 * boundary["final_boundary_clearance"]
        + 0.16 * proof["reduced_existence_support"]
    )
    switched_falsifiability = _clip01(
        0.42 * convergence["convergence_falsifiability"]
        + 0.20 * boundary["boundary_lock_confidence"]
        + 0.20 * proof["reduced_uniqueness_support"]
        + 0.18 * (1.0 - proof["reduced_proof_gap"])
    )
    switched_dependency_penalty = _clip01(
        0.56 * convergence["convergence_dependency_penalty"]
        + 0.24 * repair["repaired_reconstruction_error"]
        + 0.20 * proof["reduced_proof_gap"]
    )
    switched_identity_readiness = _clip01(
        0.30 * switched_closure
        + 0.30 * switched_falsifiability
        + 0.20 * (1.0 - switched_dependency_penalty)
        + 0.20 * boundary["boundary_lock_confidence"]
    )

    if switched_closure >= 0.75 and switched_falsifiability >= 0.77 and switched_dependency_penalty < 0.30:
        status_short = "near_first_principles_theory"
        status_label = "理论身份已经逼近第一性原理理论边界，但还差最后的严格唯一性定理"
    else:
        status_short = "phenomenological_transition"
        status_label = "理论身份继续停留在第一性原理过渡区后段"

    return {
        "headline_metrics": {
            "switched_closure": switched_closure,
            "switched_falsifiability": switched_falsifiability,
            "switched_dependency_penalty": switched_dependency_penalty,
            "switched_identity_readiness": switched_identity_readiness,
        },
        "status": {
            "status_short": status_short,
            "status_label": status_label,
        },
        "project_readout": {
            "summary": "这一轮把上下文与纤维补强、证明缺口压缩和最后边界清零尝试重新并回身份探针，判断理论是否终于逼近真正的身份切换点。",
            "next_question": "下一步需要直接针对唯一性定理本身做符号化冲刺，否则身份仍可能长期停在 near 阶段而不真正完成。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage67 Identity Switch Probe",
        "",
        f"- switched_closure: {hm['switched_closure']:.6f}",
        f"- switched_falsifiability: {hm['switched_falsifiability']:.6f}",
        f"- switched_dependency_penalty: {hm['switched_dependency_penalty']:.6f}",
        f"- switched_identity_readiness: {hm['switched_identity_readiness']:.6f}",
        f"- status_short: {status['status_short']}",
        f"- status_label: {status['status_label']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_identity_switch_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
