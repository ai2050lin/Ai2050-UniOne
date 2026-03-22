from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage68_direct_identity_assessment_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage57_task_level_repair_comparison import build_task_level_repair_comparison_summary
from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary
from stage68_direct_signal_bundle import build_direct_signal_bundle_summary
from stage68_direct_theorem_probe import build_direct_theorem_probe_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_direct_identity_assessment_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()["headline_metrics"]
    repair = build_task_level_repair_comparison_summary()["candidate_repairs"]["sqrt"]
    coeff = build_symbolic_coefficient_grounding_summary()["headline_metrics"]
    signals = build_direct_signal_bundle_summary()["headline_metrics"]
    theorem = build_direct_theorem_probe_summary()["headline_metrics"]

    direct_closure = _clip01(
        0.28 * signals["direct_structural_coherence"]
        + 0.22 * signals["direct_task_recovery_support"]
        + 0.18 * signals["direct_weight_grounding"]
        + 0.18 * theorem["direct_existence_support"]
        + 0.14 * repair["repaired_direct_structure"]
    )
    direct_falsifiability = _clip01(
        0.24 * signals["direct_boundary_resilience"]
        + 0.20 * theorem["direct_uniqueness_support"]
        + 0.18 * theorem["direct_stability_support"]
        + 0.18 * (1.0 - repair["repaired_brain_gap"])
        + 0.20 * (1.0 - repair["language_triggered_after_repair"])
    )
    direct_dependency_penalty = _clip01(
        0.28 * (1.0 - native["native_mapping_completeness"])
        + 0.22 * (1.0 - signals["direct_boundary_resilience"])
        + 0.24 * coeff["residual_grounding_gap"]
        + 0.14 * (1.0 - repair["repaired_direct_route"])
        + 0.12 * (1.0 - repair["repaired_direct_structure"])
    )
    direct_identity_readiness = _clip01(
        0.30 * direct_closure
        + 0.30 * direct_falsifiability
        + 0.20 * (1.0 - direct_dependency_penalty)
        + 0.20 * theorem["direct_theorem_readiness"]
    )

    if direct_closure >= 0.76 and direct_falsifiability >= 0.76 and direct_dependency_penalty < 0.29:
        status_short = "near_first_principles_theory"
        status_label = "即使不走嵌套闭合链，理论身份也依然逼近第一性原理理论"
    else:
        status_short = "phenomenological_transition"
        status_label = "去掉嵌套闭合链后，理论身份回到过渡区"

    return {
        "headline_metrics": {
            "direct_closure": direct_closure,
            "direct_falsifiability": direct_falsifiability,
            "direct_dependency_penalty": direct_dependency_penalty,
            "direct_identity_readiness": direct_identity_readiness,
        },
        "status": {
            "status_short": status_short,
            "status_label": status_label,
        },
        "project_readout": {
            "summary": "这一轮直接用原生量、任务量、边界量和定理探针来计算理论身份，不再套用 updated_closure、retest_closure、final_closure 这一串嵌套中间变量。",
            "next_question": "下一步要和旧链对比，确认去掉嵌套以后，结论是否仍然稳定，以及哪部分意义更强。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage68 Direct Identity Assessment",
        "",
        f"- direct_closure: {hm['direct_closure']:.6f}",
        f"- direct_falsifiability: {hm['direct_falsifiability']:.6f}",
        f"- direct_dependency_penalty: {hm['direct_dependency_penalty']:.6f}",
        f"- direct_identity_readiness: {hm['direct_identity_readiness']:.6f}",
        f"- status_short: {status['status_short']}",
        f"- status_label: {status['status_label']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_identity_assessment_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
