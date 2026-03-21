from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage60_principled_coupled_scale_repair_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage59_coupled_scale_repair import build_coupled_scale_repair_summary
from stage59_dependency_floor_search import build_dependency_floor_search_summary
from stage59_local_law_symbolic_derivation import build_local_law_symbolic_derivation_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_principled_coupled_scale_repair_summary() -> dict:
    repair = build_coupled_scale_repair_summary()
    floor = build_dependency_floor_search_summary()["headline_metrics"]
    symbolic = build_local_law_symbolic_derivation_summary()["headline_metrics"]

    base = repair["bundle_results"][repair["headline_metrics"]["best_bundle_name"]]
    principled_gain = (
        0.32 * symbolic["symbolic_closure"]
        + 0.24 * symbolic["symbolic_bridge_score"]
        + 0.18 * (1.0 - floor["dependency_floor_penalty"])
    )

    bundles = {
        "native_pressure_lift": {"dependency_relief": 0.020, "margin_gain": 0.008, "stability_gain": 0.006},
        "coefficient_grounded_route": {"dependency_relief": 0.026, "margin_gain": 0.011, "stability_gain": 0.008},
        "principled_coupled_bundle": {"dependency_relief": 0.038, "margin_gain": 0.016, "stability_gain": 0.012},
    }

    results = {}
    for name, bundle in bundles.items():
        repaired_dependency_penalty = _clip01(
            base["repaired_dependency_penalty"]
            - bundle["dependency_relief"]
            - 0.020 * principled_gain
        )
        repaired_combined_margin = _clip01(
            base["repaired_combined_margin"]
            + bundle["margin_gain"]
            + 0.012 * symbolic["symbolic_closure"]
            + 0.008 * (1.0 - floor["dependency_floor_penalty"])
        )
        repaired_update_stability = _clip01(
            base["repaired_update_stability"]
            + bundle["stability_gain"]
            + 0.010 * symbolic["symbolic_bridge_score"]
        )
        principled_repair_readiness = _clip01(
            0.36 * repaired_combined_margin
            + 0.24 * repaired_update_stability
            + 0.18 * (1.0 - repaired_dependency_penalty)
            + 0.22 * symbolic["symbolic_closure"]
        )
        principled_success = (
            repaired_combined_margin >= 0.621
            and repaired_update_stability >= 0.707
            and repaired_dependency_penalty <= 0.68
        )

        results[name] = {
            "repaired_dependency_penalty": repaired_dependency_penalty,
            "repaired_combined_margin": repaired_combined_margin,
            "repaired_update_stability": repaired_update_stability,
            "principled_repair_readiness": principled_repair_readiness,
            "principled_success": principled_success,
        }

    best_name, best_metrics = max(
        results.items(),
        key=lambda item: (item[1]["principled_success"], item[1]["principled_repair_readiness"]),
    )

    return {
        "headline_metrics": {
            "best_principled_bundle_name": best_name,
            "best_principled_dependency_penalty": best_metrics["repaired_dependency_penalty"],
            "best_principled_combined_margin": best_metrics["repaired_combined_margin"],
            "best_principled_update_stability": best_metrics["repaired_update_stability"],
            "best_principled_repair_readiness": best_metrics["principled_repair_readiness"],
            "best_principled_success": best_metrics["principled_success"],
        },
        "bundle_results": results,
        "project_readout": {
            "summary": "原理化耦合规模修复尝试把 stage59 的 repair bundle 压回符号桥与依赖地板约束之下，看是否能在降低依赖的同时保住 coupled scale stress 修复。",
            "next_question": "下一步要拿 principled bundle 继续打 dependency floor 之下的区域，验证它是不是只是换壳补丁，还是开始具备原理化特征。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage60 Principled Coupled Scale Repair",
        "",
        f"- best_principled_bundle_name: {hm['best_principled_bundle_name']}",
        f"- best_principled_dependency_penalty: {hm['best_principled_dependency_penalty']:.6f}",
        f"- best_principled_combined_margin: {hm['best_principled_combined_margin']:.6f}",
        f"- best_principled_update_stability: {hm['best_principled_update_stability']:.6f}",
        f"- best_principled_repair_readiness: {hm['best_principled_repair_readiness']:.6f}",
        f"- best_principled_success: {hm['best_principled_success']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_principled_coupled_scale_repair_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
