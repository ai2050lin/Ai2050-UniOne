from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage65_first_principles_identity_final_probe_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage61_theory_identity_retest import build_theory_identity_retest_summary
from stage65_boundary_to_completion_lock import build_boundary_to_completion_lock_summary
from stage65_completion_gap_attack import build_completion_gap_attack_summary
from stage65_selector_master_equation_closure import build_selector_master_equation_closure_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_first_principles_identity_final_probe_summary() -> dict:
    retest = build_theory_identity_retest_summary()["headline_metrics"]
    lock = build_boundary_to_completion_lock_summary()["headline_metrics"]
    gap = build_completion_gap_attack_summary()["headline_metrics"]
    master = build_selector_master_equation_closure_summary()["headline_metrics"]

    final_closure = _clip01(
        0.34 * retest["retest_closure"]
        + 0.28 * master["master_equation_closure"]
        + 0.22 * lock["completion_lock_score"]
        + 0.16 * (1.0 - gap["attacked_completion_gap"])
    )
    final_falsifiability = _clip01(
        0.26 * retest["retest_falsifiability"]
        + 0.28 * master["equation_constraint_lock"]
        + 0.24 * lock["completion_lock_confidence"]
        + 0.22 * (1.0 - gap["residual_completion_blocker"])
    )
    final_dependency_penalty = _clip01(
        0.30 * retest["retest_dependency_penalty"]
        + 0.30 * gap["residual_completion_blocker"]
        + 0.20 * master["residual_master_gap"]
        + 0.20 * (1.0 - lock["completion_lock_confidence"])
    )
    final_identity_readiness = _clip01(
        0.30 * final_closure
        + 0.30 * final_falsifiability
        + 0.20 * (1.0 - final_dependency_penalty)
        + 0.20 * lock["completion_lock_confidence"]
    )

    if final_closure >= 0.72 and final_falsifiability >= 0.78 and final_dependency_penalty < 0.34:
        status_short = "near_first_principles_theory"
        status_label = "已逼近第一性原理理论边界，但仍差最终闭合"
    else:
        status_short = "phenomenological_transition"
        status_label = "仍属唯象模型，但处于第一性原理过渡区后段"

    return {
        "headline_metrics": {
            "final_closure": final_closure,
            "final_falsifiability": final_falsifiability,
            "final_dependency_penalty": final_dependency_penalty,
            "final_identity_readiness": final_identity_readiness,
        },
        "status": {
            "status_short": status_short,
            "status_label": status_label,
        },
        "project_readout": {
            "summary": "最终身份探针把主方程闭合、完成缺口攻击和边界锁定结果重新合成，判断当前理论身份是否已逼近第一性原理理论边界。",
            "next_question": "下一步要针对最后一步主方程闭合做最终冲刺，否则身份仍只会停在逼近边界而非真正跨越。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage65 First Principles Identity Final Probe",
        "",
        f"- final_closure: {hm['final_closure']:.6f}",
        f"- final_falsifiability: {hm['final_falsifiability']:.6f}",
        f"- final_dependency_penalty: {hm['final_dependency_penalty']:.6f}",
        f"- final_identity_readiness: {hm['final_identity_readiness']:.6f}",
        f"- status_short: {status['status_short']}",
        f"- status_label: {status['status_label']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_first_principles_identity_final_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
