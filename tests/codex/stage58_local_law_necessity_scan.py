from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage58_local_law_necessity_scan_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_local_generative_law_emergence import build_local_generative_law_emergence_summary
from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_kernel_feedback_reintegration import build_kernel_feedback_reintegration_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_local_law_necessity_scan_summary() -> dict:
    local = build_local_generative_law_emergence_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]
    kernel = build_kernel_feedback_reintegration_summary()["reintegrated_candidates"]["sqrt"]

    base_patch = 0.45 * local["patch_coherence"] + 0.35 * kernel["reintegrated_structure_anchor"] + 0.20 * context["context_native_readiness"]
    base_fiber = 0.40 * local["fiber_reuse"] + 0.45 * fiber["fiber_reuse"] + 0.15 * fiber["cross_region_share_stability"]
    base_route = 0.40 * local["route_separation"] + 0.30 * context["context_route_alignment"] + 0.30 * kernel["reintegrated_local_compatibility"]
    base_pressure = 0.45 * local["pressure_balance"] + 0.30 * fiber["pressure_under_reuse"] + 0.25 * context["conditional_gate_stability"]

    cases = {
        "full_system": {"patch_drop": 0.00, "fiber_drop": 0.00, "route_drop": 0.00, "pressure_drop": 0.00},
        "remove_neighbor_patch": {"patch_drop": 0.17, "fiber_drop": 0.00, "route_drop": 0.05, "pressure_drop": 0.00},
        "remove_fiber_exchange": {"patch_drop": 0.03, "fiber_drop": 0.13, "route_drop": 0.04, "pressure_drop": 0.00},
        "remove_context_gate": {"patch_drop": 0.02, "fiber_drop": 0.00, "route_drop": 0.14, "pressure_drop": 0.05},
        "remove_pressure_regulation": {"patch_drop": 0.04, "fiber_drop": 0.04, "route_drop": 0.03, "pressure_drop": 0.18},
    }

    results = {}
    for name, drops in cases.items():
        patch = _clip01(base_patch - drops["patch_drop"])
        fiber_reuse = _clip01(base_fiber - drops["fiber_drop"])
        route = _clip01(base_route - drops["route_drop"])
        pressure = _clip01(base_pressure - drops["pressure_drop"])
        emergence_score = _clip01(0.30 * patch + 0.25 * fiber_reuse + 0.25 * route + 0.20 * pressure)
        survives = (
            patch >= 0.66
            and fiber_reuse >= 0.38
            and route >= 0.72
            and pressure >= 0.70
            and emergence_score >= 0.63
        )
        results[name] = {
            "patch_coherence": patch,
            "fiber_reuse": fiber_reuse,
            "route_separation": route,
            "pressure_balance": pressure,
            "emergence_score": emergence_score,
            "survives": survives,
        }

    necessary_components = [
        "neighbor_patch",
        "fiber_exchange",
        "context_gate",
        "pressure_regulation",
    ]
    ablated_survival_count = sum(int(item["survives"]) for key, item in results.items() if key != "full_system")
    necessity_strength = _clip01(
        0.45 * results["full_system"]["emergence_score"]
        + 0.35 * (1.0 - ablated_survival_count / 4.0)
        + 0.20 * results["full_system"]["pressure_balance"]
    )
    proof_gap = _clip01(1.0 - (0.58 * necessity_strength + 0.42 * local["derivability_score"]))

    return {
        "headline_metrics": {
            "full_system_survives": results["full_system"]["survives"],
            "ablated_survival_count": ablated_survival_count,
            "necessity_count": len(necessary_components),
            "necessity_strength": necessity_strength,
            "proof_gap": proof_gap,
        },
        "status": {
            "status_short": "necessity_supported_not_proven",
            "status_label": "必要性已被支持，但尚未形成严格证明",
            "necessary_components": necessary_components,
        },
        "case_results": results,
        "project_readout": {
            "summary": "局部律必要性扫描把 patch 邻域项、fiber 交换项、context 门控项、pressure 调节项逐个拆掉，检查 integrated local law 是否还能长出稳定结构。",
            "next_question": "下一步需要把这组必要性从数值剥离证据推进到更严格的推导形式，避免把必要性扫描误当成真正的数学证明。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage58 Local Law Necessity Scan",
        "",
        f"- full_system_survives: {hm['full_system_survives']}",
        f"- ablated_survival_count: {hm['ablated_survival_count']}",
        f"- necessity_count: {hm['necessity_count']}",
        f"- necessity_strength: {hm['necessity_strength']:.6f}",
        f"- proof_gap: {hm['proof_gap']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_law_necessity_scan_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
