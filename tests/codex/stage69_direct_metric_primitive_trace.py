from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage69_direct_metric_primitive_trace_20260322"
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
from stage68_direct_identity_assessment import build_direct_identity_assessment_summary
from stage68_direct_signal_bundle import build_direct_signal_bundle_summary
from stage68_direct_theorem_probe import build_direct_theorem_probe_summary


def build_direct_metric_primitive_trace_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()
    native_hm = native["headline_metrics"]
    mapping = native["candidate_mapping"]
    context = build_context_native_grounding_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    repair = build_task_level_repair_comparison_summary()["candidate_repairs"]["sqrt"]
    coeff = build_symbolic_coefficient_grounding_summary()["headline_metrics"]
    uniq = build_coefficient_uniqueness_probe_summary()["headline_metrics"]
    band = build_low_dependency_band_stress_summary()["headline_metrics"]
    signals = build_direct_signal_bundle_summary()["headline_metrics"]
    theorem = build_direct_theorem_probe_summary()["headline_metrics"]
    direct = build_direct_identity_assessment_summary()["headline_metrics"]

    primitive_origin = {
        "P_patch": mapping["P_patch"]["native_variable_candidate"],
        "F_fiber": mapping["F_fiber"]["native_variable_candidate"],
        "R_route": mapping["R_route"]["native_variable_candidate"],
        "C_context": mapping["C_context"]["native_variable_candidate"],
        "L_plasticity": mapping["L_plasticity"]["native_variable_candidate"],
        "Pi_pressure": mapping["Pi_pressure"]["native_variable_candidate"],
    }

    signal_breakdown = {
        "direct_structural_coherence": {
            "0.18*native_mapping_completeness": 0.18 * native_hm["native_mapping_completeness"],
            "0.16*context_native_readiness": 0.16 * context["context_native_readiness"],
            "0.10*conditional_gate_stability": 0.10 * context["conditional_gate_stability"],
            "0.10*fiber_reuse": 0.10 * fiber["fiber_reuse"],
            "0.10*cross_region_share_stability": 0.10 * fiber["cross_region_share_stability"],
            "0.18*native_coefficient_score": 0.18 * coeff["native_coefficient_score"],
            "0.18*repaired_direct_structure": 0.18 * repair["repaired_direct_structure"],
            "sum": signals["direct_structural_coherence"],
        },
        "direct_task_recovery_support": {
            "0.18*(1-long_forgetting)": 0.18 * (1.0 - repair["repaired_long_forgetting"]),
            "0.14*(1-perplexity_ratio)": 0.14 * (1.0 - min(1.0, repair["repaired_base_perplexity_delta"] / 1200.0)),
            "0.16*novel_accuracy": 0.16 * repair["repaired_novel_accuracy_after"],
            "0.16*direct_structure": 0.16 * repair["repaired_direct_structure"],
            "0.14*direct_route": 0.14 * repair["repaired_direct_route"],
            "0.10*shared_red_reuse": 0.10 * repair["repaired_shared_red_reuse"],
            "0.12*(1-brain_gap)": 0.12 * (1.0 - repair["repaired_brain_gap"]),
            "sum": signals["direct_task_recovery_support"],
        },
        "direct_boundary_resilience": {
            "0.42*band_resilience_score": 0.42 * band["band_resilience_score"],
            "0.28*(safe_points/5)": 0.28 * (band["stressed_safe_point_count"] / 5.0),
            "0.16*repaired_direct_route": 0.16 * repair["repaired_direct_route"],
            "0.14*(1-brain_gap)": 0.14 * (1.0 - repair["repaired_brain_gap"]),
            "sum": signals["direct_boundary_resilience"],
        },
        "direct_weight_grounding": {
            "0.30*coefficient_grounding_coverage": 0.30 * coeff["coefficient_grounding_coverage"],
            "0.28*native_coefficient_score": 0.28 * coeff["native_coefficient_score"],
            "0.20*(1-residual_grounding_gap)": 0.20 * (1.0 - coeff["residual_grounding_gap"]),
            "0.12*shared_constraints": 0.12 * uniq["shared_constraints"],
            "0.10*language_brain_agreement": 0.10 * uniq["language_brain_agreement"],
            "sum": signals["direct_weight_grounding"],
        },
    }

    identity_breakdown = {
        "direct_closure": {
            "0.28*direct_structural_coherence": 0.28 * signals["direct_structural_coherence"],
            "0.22*direct_task_recovery_support": 0.22 * signals["direct_task_recovery_support"],
            "0.18*direct_weight_grounding": 0.18 * signals["direct_weight_grounding"],
            "0.18*direct_existence_support": 0.18 * theorem["direct_existence_support"],
            "0.14*repaired_direct_structure": 0.14 * repair["repaired_direct_structure"],
            "sum": direct["direct_closure"],
        },
        "direct_falsifiability": {
            "0.24*direct_boundary_resilience": 0.24 * signals["direct_boundary_resilience"],
            "0.20*direct_uniqueness_support": 0.20 * theorem["direct_uniqueness_support"],
            "0.18*direct_stability_support": 0.18 * theorem["direct_stability_support"],
            "0.18*(1-brain_gap)": 0.18 * (1.0 - repair["repaired_brain_gap"]),
            "0.20*(1-language_triggered)": 0.20 * (1.0 - float(repair["language_triggered_after_repair"])),
            "sum": direct["direct_falsifiability"],
        },
        "direct_dependency_penalty": {
            "0.28*(1-native_mapping_completeness)": 0.28 * (1.0 - native_hm["native_mapping_completeness"]),
            "0.22*(1-direct_boundary_resilience)": 0.22 * (1.0 - signals["direct_boundary_resilience"]),
            "0.24*residual_grounding_gap": 0.24 * coeff["residual_grounding_gap"],
            "0.14*(1-repaired_direct_route)": 0.14 * (1.0 - repair["repaired_direct_route"]),
            "0.12*(1-repaired_direct_structure)": 0.12 * (1.0 - repair["repaired_direct_structure"]),
            "sum": direct["direct_dependency_penalty"],
        },
        "direct_identity_readiness": {
            "0.30*direct_closure": 0.30 * direct["direct_closure"],
            "0.30*direct_falsifiability": 0.30 * direct["direct_falsifiability"],
            "0.20*(1-direct_dependency_penalty)": 0.20 * (1.0 - direct["direct_dependency_penalty"]),
            "0.20*direct_theorem_readiness": 0.20 * theorem["direct_theorem_readiness"],
            "sum": direct["direct_identity_readiness"],
        },
    }

    return {
        "primitive_origin": primitive_origin,
        "signal_breakdown": signal_breakdown,
        "identity_breakdown": identity_breakdown,
        "status": {
            "status_short": "direct_metric_trace_ready",
            "status_label": "四个 direct 指标已经被完整追溯到原生变量、任务量和系数量来源",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# Stage69 Direct Metric Primitive Trace",
        "",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_metric_primitive_trace_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
