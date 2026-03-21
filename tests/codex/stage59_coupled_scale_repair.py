from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage59_coupled_scale_repair_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage58_counterexample_priority_probe import build_counterexample_priority_probe_summary
from stage58_large_scale_long_horizon_bundle import build_large_scale_long_horizon_bundle_summary
from stage58_repair_dependency_reduction import build_repair_dependency_reduction_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_coupled_scale_repair_summary() -> dict:
    horizon = build_large_scale_long_horizon_bundle_summary()
    reduction = build_repair_dependency_reduction_summary()
    probe = build_counterexample_priority_probe_summary()["headline_metrics"]

    best_name = reduction["headline_metrics"]["best_strategy_name"]
    best = reduction["strategy_results"][best_name]
    base_case = horizon["case_results"]["coupled_scale_stress"]
    base_fatigue = horizon["headline_metrics"]["fatigue"]
    base_update = horizon["case_results"]["continual_online_update"]["update_stability"]
    base_language = horizon["case_results"]["long_context_persistence"]["language_keep"]
    base_structure = horizon["case_results"]["cross_region_brain_bridge"]["structure_keep"]

    bundles = {
        "pressure_only": {"pressure_gain": 0.024, "route_gain": 0.010, "scale_gain": 0.008, "dependency_cost": 0.010},
        "context_pressure_weave": {"pressure_gain": 0.032, "route_gain": 0.018, "scale_gain": 0.012, "dependency_cost": 0.018},
        "route_pressure_weave": {"pressure_gain": 0.028, "route_gain": 0.022, "scale_gain": 0.014, "dependency_cost": 0.015},
        "coupled_scale_bundle": {"pressure_gain": 0.040, "route_gain": 0.028, "scale_gain": 0.020, "dependency_cost": 0.020},
    }

    results = {}
    for name, bundle in bundles.items():
        fatigue_reduction = (
            0.50 * bundle["pressure_gain"]
            + 0.22 * bundle["route_gain"]
            + 0.18 * bundle["scale_gain"]
            + 0.03 * (1.0 - best["reduced_dependency_penalty"])
        )
        repaired_fatigue = _clip01(base_fatigue - fatigue_reduction)
        combined_margin = _clip01(
            base_case["combined_margin"]
            + 0.55 * bundle["pressure_gain"]
            + 0.40 * bundle["route_gain"]
            + 0.35 * bundle["scale_gain"]
            + 0.05 * (1.0 - repaired_fatigue)
            - 0.05 * probe["closure_risk_index"]
            - 0.02 * bundle["dependency_cost"]
        )
        repaired_update_stability = _clip01(
            base_update
            + 0.45 * bundle["pressure_gain"]
            + 0.35 * bundle["route_gain"]
            + 0.10 * bundle["scale_gain"]
            - 0.03 * bundle["dependency_cost"]
        )
        repaired_language_keep = _clip01(
            base_language
            + 0.22 * bundle["pressure_gain"]
            + 0.18 * bundle["route_gain"]
            + 0.08 * bundle["scale_gain"]
            - 0.02 * bundle["dependency_cost"]
        )
        repaired_structure_keep = _clip01(
            base_structure
            + 0.12 * bundle["pressure_gain"]
            + 0.26 * bundle["route_gain"]
            + 0.14 * bundle["scale_gain"]
            - 0.03 * bundle["dependency_cost"]
        )
        repaired_dependency_penalty = _clip01(best["reduced_dependency_penalty"] + bundle["dependency_cost"])
        repair_success = (
            combined_margin >= 0.61
            and repaired_update_stability >= 0.69
            and repaired_language_keep >= 0.78
            and repaired_structure_keep >= 0.71
        )
        repair_readiness = _clip01(
            0.34 * combined_margin
            + 0.20 * repaired_update_stability
            + 0.16 * repaired_language_keep
            + 0.16 * repaired_structure_keep
            + 0.14 * (1.0 - repaired_dependency_penalty)
        )

        results[name] = {
            "repaired_fatigue": repaired_fatigue,
            "repaired_combined_margin": combined_margin,
            "repaired_update_stability": repaired_update_stability,
            "repaired_language_keep": repaired_language_keep,
            "repaired_structure_keep": repaired_structure_keep,
            "repaired_dependency_penalty": repaired_dependency_penalty,
            "repair_success": repair_success,
            "repair_readiness": repair_readiness,
        }

    best_bundle_name, best_bundle = max(
        results.items(),
        key=lambda item: (item[1]["repair_success"], item[1]["repair_readiness"]),
    )

    return {
        "headline_metrics": {
            "best_bundle_name": best_bundle_name,
            "base_combined_margin": base_case["combined_margin"],
            "best_repaired_combined_margin": best_bundle["repaired_combined_margin"],
            "best_repaired_dependency_penalty": best_bundle["repaired_dependency_penalty"],
            "best_repair_readiness": best_bundle["repair_readiness"],
            "best_repair_success": best_bundle["repair_success"],
        },
        "bundle_results": results,
        "project_readout": {
            "summary": "耦合规模压力修复轮专门针对长时程组合反例，不再平均修补所有指标，而是比较不同 repair bundle 谁能把 coupled scale stress 拉回安全区。",
            "next_question": "下一步要把最优 bundle 固定成 replay 基线，再继续向下搜索 dependency floor，判断还能把显式依赖再压多少。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage59 Coupled Scale Repair",
        "",
        f"- best_bundle_name: {hm['best_bundle_name']}",
        f"- base_combined_margin: {hm['base_combined_margin']:.6f}",
        f"- best_repaired_combined_margin: {hm['best_repaired_combined_margin']:.6f}",
        f"- best_repaired_dependency_penalty: {hm['best_repaired_dependency_penalty']:.6f}",
        f"- best_repair_readiness: {hm['best_repair_readiness']:.6f}",
        f"- best_repair_success: {hm['best_repair_success']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_coupled_scale_repair_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
