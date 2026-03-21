from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_theory_status_assessment_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_first_principles_transition_framework import build_first_principles_transition_framework_summary
from stage56_local_generative_law_emergence import build_local_generative_law_emergence_summary
from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage57_failure_boundary_trigger import build_failure_boundary_trigger_summary
from stage57_minimal_repair_set_ablation import build_minimal_repair_set_ablation_summary
from stage57_task_level_repair_comparison import build_task_level_repair_comparison_summary


def build_theory_status_assessment_summary() -> dict:
    fp = build_first_principles_transition_framework_summary()["headline_metrics"]
    local = build_local_generative_law_emergence_summary()["headline_metrics"]
    native = build_native_variable_candidate_mapping_summary()["headline_metrics"]
    failure = build_failure_boundary_trigger_summary()["headline_metrics"]
    repair = build_task_level_repair_comparison_summary()["headline_metrics"]
    ablation = build_minimal_repair_set_ablation_summary()

    necessary_count = len(ablation["necessary_components"])
    repair_dependency_penalty = min(1.0, 0.30 + 0.25 * necessary_count + (0.20 if ablation["headline_metrics"]["minimum_joint_repair_required"] else 0.0))

    phenomenology_strength = max(
        0.0,
        min(
            1.0,
            0.28 * repair["best_repair_readiness"]
            + 0.24 * local["local_law_emergence_score"]
            + 0.24 * native["primitive_set_readiness"]
            + 0.24 * fp["primitive_transition_readiness"],
        ),
    )
    first_principles_support = max(
        0.0,
        min(
            1.0,
            0.24 * fp["first_principles_transition_score"]
            + 0.18 * fp["local_law_closure"]
            + 0.18 * native["native_mapping_completeness"]
            + 0.20 * failure["boundary_system_readiness"]
            + 0.20 * repair["best_repair_readiness"],
        ),
    )
    first_principles_closure = max(
        0.0,
        min(1.0, first_principles_support - 0.22 * repair_dependency_penalty - 0.12 * (1.0 - local["derivability_score"])),
    )
    falsifiability_strength = max(
        0.0,
        min(
            1.0,
            0.35 * failure["boundary_system_readiness"]
            + 0.25 * repair["best_repair_task_count"] / 2.0
            + 0.20 * (1.0 - repair_dependency_penalty)
            + 0.20 * fp["falsifiability_upgrade"],
        ),
    )

    if first_principles_closure >= 0.75 and falsifiability_strength >= 0.75 and repair_dependency_penalty < 0.40:
        status_label = "基于第一性原理的理论"
        status_short = "first_principles_theory"
    elif first_principles_closure >= 0.50 and falsifiability_strength >= 0.70:
        status_label = "仍属唯象模型，但已进入第一性原理过渡区"
        status_short = "phenomenological_transition"
    else:
        status_label = "唯象模型"
        status_short = "phenomenological_model"

    return {
        "headline_metrics": {
            "phenomenology_strength": phenomenology_strength,
            "first_principles_support": first_principles_support,
            "first_principles_closure": first_principles_closure,
            "falsifiability_strength": falsifiability_strength,
            "repair_dependency_penalty": repair_dependency_penalty,
        },
        "status": {
            "status_short": status_short,
            "status_label": status_label,
            "necessary_components": ablation["necessary_components"],
        },
        "project_readout": {
            "summary": "理论状态评估把第一性原理过渡、原生变量映射、局部生成律可导出性、可判伪性、修复依赖度合并成一个总判断，用来确定当前体系仍是唯象模型，还是已经真正跨入第一性原理理论。",
            "next_question": "下一步必须继续降低修复依赖、提升原生变量到局部生成律的可导出性；只有在削弱补丁后仍能保住任务层修复，理论才可能更接近真正的第一性原理状态。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage57 Theory Status Assessment",
        "",
        f"- status_short: {status['status_short']}",
        f"- status_label: {status['status_label']}",
        f"- phenomenology_strength: {hm['phenomenology_strength']:.6f}",
        f"- first_principles_support: {hm['first_principles_support']:.6f}",
        f"- first_principles_closure: {hm['first_principles_closure']:.6f}",
        f"- falsifiability_strength: {hm['falsifiability_strength']:.6f}",
        f"- repair_dependency_penalty: {hm['repair_dependency_penalty']:.6f}",
        f"- necessary_components: {', '.join(status['necessary_components'])}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_theory_status_assessment_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
