from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage58_large_scale_long_horizon_bundle_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_large_scale_long_context_online_validation import build_large_scale_long_context_online_validation_summary
from stage57_real_boundary_stress_generator import build_real_boundary_stress_generator_summary
from stage58_repair_dependency_reduction import build_repair_dependency_reduction_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_scale_long_horizon_bundle_summary() -> dict:
    scale = build_large_scale_long_context_online_validation_summary()["headline_metrics"]
    stress = build_real_boundary_stress_generator_summary()
    reduction = build_repair_dependency_reduction_summary()

    best_name = reduction["headline_metrics"]["best_strategy_name"]
    best = reduction["strategy_results"][best_name]
    fatigue = (
        0.25 * scale["scale_collapse_risk"]
        + 0.20 * scale["scale_forgetting_penalty"]
        + 0.18 * best["reduced_dependency_penalty"]
        + 0.12 * (1.0 - scale["scale_novel_gain"])
    )

    cases = {
        "long_context_persistence": {
            "language_keep": _clip01(
                scale["scale_language_keep"]
                + 0.08 * best["repaired_novel_accuracy_after"]
                - 0.35 * fatigue
            ),
            "route_retention": _clip01(
                scale["long_context_generalization"]
                + 0.05 * best["context_support"]
                - 0.25 * fatigue
            ),
        },
        "cross_region_brain_bridge": {
            "structure_keep": _clip01(
                scale["scale_structure_keep"]
                + 0.12 * best["repaired_direct_structure"]
                - 0.18 * fatigue
            ),
            "shared_reuse": _clip01(best["repaired_shared_red_reuse"] - 0.14 * fatigue),
        },
        "continual_online_update": {
            "update_stability": _clip01(
                0.55 * best["reduced_repair_readiness"]
                + 0.20 * (1.0 - stress["scale_metrics"]["scale_collapse_risk"])
                + 0.15 * (1.0 - best["reduced_dependency_penalty"])
                + 0.10 * (1.0 - fatigue)
            ),
        },
        "coupled_scale_stress": {
            "combined_margin": _clip01(
                scale["scale_readiness"]
                + 0.18 * best["reduced_repair_readiness"]
                - 0.25 * best["reduced_dependency_penalty"]
                - 0.18 * scale["scale_collapse_risk"]
                - 0.10 * (1.0 - scale["scale_novel_gain"])
            ),
        },
    }

    case_results = {
        "long_context_persistence": {
            **cases["long_context_persistence"],
            "triggered": (
                cases["long_context_persistence"]["language_keep"] < 0.73
                or cases["long_context_persistence"]["route_retention"] < 0.60
            ),
        },
        "cross_region_brain_bridge": {
            **cases["cross_region_brain_bridge"],
            "triggered": (
                cases["cross_region_brain_bridge"]["structure_keep"] < 0.68
                or cases["cross_region_brain_bridge"]["shared_reuse"] < 0.79
            ),
        },
        "continual_online_update": {
            **cases["continual_online_update"],
            "triggered": cases["continual_online_update"]["update_stability"] < 0.64,
        },
        "coupled_scale_stress": {
            **cases["coupled_scale_stress"],
            "triggered": cases["coupled_scale_stress"]["combined_margin"] < 0.58,
        },
    }

    validated_case_count = sum(int(not item["triggered"]) for item in case_results.values())
    survival_rate = validated_case_count / len(case_results)
    worst_case_name, _worst = min(
        case_results.items(),
        key=lambda item: item[1].get(
            "combined_margin",
            item[1].get("update_stability", item[1].get("structure_keep", item[1].get("language_keep", 1.0))),
        ),
    )
    large_scale_long_horizon_readiness = _clip01(
        0.30 * survival_rate
        + 0.25 * scale["scale_readiness"]
        + 0.20 * best["reduced_repair_readiness"]
        + 0.15 * (1.0 - fatigue)
        + 0.10 * (1.0 - best["reduced_dependency_penalty"])
    )

    return {
        "headline_metrics": {
            "best_strategy_name": best_name,
            "validated_case_count": validated_case_count,
            "survival_rate": survival_rate,
            "fatigue": fatigue,
            "large_scale_long_horizon_readiness": large_scale_long_horizon_readiness,
            "worst_case_name": worst_case_name,
        },
        "case_results": case_results,
        "project_readout": {
            "summary": "更大规模长期验证轮把 dependency reduction 的最优策略放回长上下文、跨区域桥接、持续更新和耦合规模压力下，检查当前修复集在更长时程是否还能站住。",
            "next_question": "下一步要优先处理 coupled scale stress，因为它是当前唯一仍然把系统重新压回危险区的长期组合反例。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage58 Large Scale Long Horizon Bundle",
        "",
        f"- best_strategy_name: {hm['best_strategy_name']}",
        f"- validated_case_count: {hm['validated_case_count']}",
        f"- survival_rate: {hm['survival_rate']:.6f}",
        f"- fatigue: {hm['fatigue']:.6f}",
        f"- large_scale_long_horizon_readiness: {hm['large_scale_long_horizon_readiness']:.6f}",
        f"- worst_case_name: {hm['worst_case_name']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_scale_long_horizon_bundle_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
