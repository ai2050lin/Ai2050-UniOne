from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage58_counterexample_priority_probe_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_brain_bridge_boundary_trigger import build_brain_bridge_boundary_trigger_summary
from stage57_language_task_boundary_trigger import build_language_task_boundary_trigger_summary
from stage57_real_boundary_stress_generator import build_real_boundary_stress_generator_summary
from stage58_large_scale_long_horizon_bundle import build_large_scale_long_horizon_bundle_summary
from stage58_local_law_necessity_scan import build_local_law_necessity_scan_summary
from stage58_repair_dependency_reduction import build_repair_dependency_reduction_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_counterexample_priority_probe_summary() -> dict:
    language = build_language_task_boundary_trigger_summary()
    brain = build_brain_bridge_boundary_trigger_summary()
    stress = build_real_boundary_stress_generator_summary()
    reduction = build_repair_dependency_reduction_summary()
    horizon = build_large_scale_long_horizon_bundle_summary()
    necessity = build_local_law_necessity_scan_summary()["headline_metrics"]

    best = reduction["strategy_results"][reduction["headline_metrics"]["best_strategy_name"]]

    scenarios = {
        "context_overload": {
            "triggered": stress["scenario_results"]["context_overload"]["triggered"],
            "risk_score": _clip01(
                0.35 * language["headline_metrics"]["stressed_long_forgetting"]
                + 0.20 * (1.0 - stress["scenario_results"]["context_overload"]["context_route_alignment"])
                + 0.20 * stress["scale_metrics"]["scale_collapse_risk"]
                + 0.25 * (1.0 - best["context_support"])
            ),
        },
        "fiber_congestion_wave": {
            "triggered": stress["scenario_results"]["fiber_congestion_wave"]["triggered"],
            "risk_score": _clip01(
                0.30 * (1.0 - stress["scenario_results"]["fiber_congestion_wave"]["fiber_reuse"])
                + 0.20 * (1.0 - stress["scenario_results"]["fiber_congestion_wave"]["route_fiber_coupling_balance"])
                + 0.25 * (1.0 - brain["headline_metrics"]["stressed_direct_structure"])
                + 0.25 * (1.0 - best["fiber_support"])
            ),
        },
        "kernel_domination_rebound": {
            "triggered": stress["scenario_results"]["kernel_domination_rebound"]["triggered"],
            "risk_score": _clip01(
                0.40 * stress["scenario_results"]["kernel_domination_rebound"]["domination_penalty"]
                + 0.30 * best["reduced_dependency_penalty"]
                + 0.30 * (1.0 - best["reduced_repair_readiness"])
            ),
        },
        "coupled_patch_erosion": {
            "triggered": stress["scenario_results"]["coupled_patch_erosion"]["triggered"],
            "risk_score": _clip01(
                0.30 * (1.0 - stress["scenario_results"]["coupled_patch_erosion"]["reintegrated_structure_anchor"])
                + 0.25 * (1.0 - stress["scenario_results"]["coupled_patch_erosion"]["reintegrated_local_compatibility"])
                + 0.20 * stress["scale_metrics"]["scale_collapse_risk"]
                + 0.25 * necessity["proof_gap"]
            ),
        },
        "long_horizon_coupled_scale_stress": {
            "triggered": horizon["case_results"]["coupled_scale_stress"]["triggered"],
            "risk_score": _clip01(
                0.40 * (1.0 - horizon["case_results"]["coupled_scale_stress"]["combined_margin"])
                + 0.30 * best["reduced_dependency_penalty"]
                + 0.20 * stress["scale_metrics"]["scale_collapse_risk"]
                + 0.20 * (1.0 - horizon["headline_metrics"]["survival_rate"])
            ),
        },
    }

    top_name, top_case = max(scenarios.items(), key=lambda item: item[1]["risk_score"])
    probe_coverage = sum(1 for item in scenarios.values() if item["triggered"]) / len(scenarios)
    closure_risk_index = _clip01(
        0.50 * top_case["risk_score"]
        + 0.25 * (1.0 - horizon["headline_metrics"]["survival_rate"])
        + 0.20 * best["reduced_dependency_penalty"]
        + 0.10 * necessity["proof_gap"]
    )

    return {
        "headline_metrics": {
            "top_priority_name": top_name,
            "top_priority_risk_score": top_case["risk_score"],
            "top_priority_triggered": top_case["triggered"],
            "probe_coverage": probe_coverage,
            "closure_risk_index": closure_risk_index,
        },
        "scenario_priorities": scenarios,
        "project_readout": {
            "summary": "反例优先轮把当前已经出现的真实失败家族按 closure risk 重新排序，强迫项目先处理最可能再次击穿主核的长期耦合反例。",
            "next_question": "下一步应先围绕 long horizon coupled scale stress 设计专门修复，而不是继续均匀打磨所有局部指标。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage58 Counterexample Priority Probe",
        "",
        f"- top_priority_name: {hm['top_priority_name']}",
        f"- top_priority_risk_score: {hm['top_priority_risk_score']:.6f}",
        f"- top_priority_triggered: {hm['top_priority_triggered']}",
        f"- probe_coverage: {hm['probe_coverage']:.6f}",
        f"- closure_risk_index: {hm['closure_risk_index']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_counterexample_priority_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
