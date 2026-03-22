from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage68_direct_signal_bundle_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_task_level_repair_comparison import build_task_level_repair_comparison_summary
from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary
from stage61_coefficient_uniqueness_probe import build_coefficient_uniqueness_probe_summary
from stage62_low_dependency_band_stress import build_low_dependency_band_stress_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_direct_signal_bundle_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    repair = build_task_level_repair_comparison_summary()["candidate_repairs"]["sqrt"]
    coeff = build_symbolic_coefficient_grounding_summary()["headline_metrics"]
    uniq = build_coefficient_uniqueness_probe_summary()["headline_metrics"]
    band = build_low_dependency_band_stress_summary()["headline_metrics"]

    direct_structural_coherence = _clip01(
        0.18 * native["native_mapping_completeness"]
        + 0.16 * context["context_native_readiness"]
        + 0.10 * context["conditional_gate_stability"]
        + 0.10 * fiber["fiber_reuse"]
        + 0.10 * fiber["cross_region_share_stability"]
        + 0.18 * coeff["native_coefficient_score"]
        + 0.18 * repair["repaired_direct_structure"]
    )
    direct_task_recovery_support = _clip01(
        0.18 * (1.0 - repair["repaired_long_forgetting"])
        + 0.14 * (1.0 - min(1.0, repair["repaired_base_perplexity_delta"] / 1200.0))
        + 0.16 * repair["repaired_novel_accuracy_after"]
        + 0.16 * repair["repaired_direct_structure"]
        + 0.14 * repair["repaired_direct_route"]
        + 0.10 * repair["repaired_shared_red_reuse"]
        + 0.12 * (1.0 - repair["repaired_brain_gap"])
    )
    direct_boundary_resilience = _clip01(
        0.42 * band["band_resilience_score"]
        + 0.28 * (band["stressed_safe_point_count"] / 5.0)
        + 0.16 * repair["repaired_direct_route"]
        + 0.14 * (1.0 - repair["repaired_brain_gap"])
    )
    direct_weight_grounding = _clip01(
        0.30 * coeff["coefficient_grounding_coverage"]
        + 0.28 * coeff["native_coefficient_score"]
        + 0.20 * (1.0 - coeff["residual_grounding_gap"])
        + 0.12 * uniq["shared_constraints"]
        + 0.10 * uniq["language_brain_agreement"]
    )

    return {
        "headline_metrics": {
            "direct_structural_coherence": direct_structural_coherence,
            "direct_task_recovery_support": direct_task_recovery_support,
            "direct_boundary_resilience": direct_boundary_resilience,
            "direct_weight_grounding": direct_weight_grounding,
        },
        "status": {
            "status_short": "direct_signal_bundle_ready",
            "status_label": "最终身份判断所需的核心信号已经被压回更底层、可解释的直接量",
        },
        "project_readout": {
            "summary": "这一轮不再使用 updated_closure 或 retest_closure，而是直接从原生变量、任务修复、边界韧性、系数落地四类底层信号出发构造最终判断。",
            "next_question": "下一步要把这些直接信号进一步压成直接的存在性、唯一性、稳定性探针，看看是否还能维持和旧链相近的结论。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage68 Direct Signal Bundle",
        "",
        f"- direct_structural_coherence: {hm['direct_structural_coherence']:.6f}",
        f"- direct_task_recovery_support: {hm['direct_task_recovery_support']:.6f}",
        f"- direct_boundary_resilience: {hm['direct_boundary_resilience']:.6f}",
        f"- direct_weight_grounding: {hm['direct_weight_grounding']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_signal_bundle_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
