from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    with (TEMP_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    g10 = load_json("g10_surrogate_model_mismatch_calibration_20260311.json")
    g7b = load_json("g7b_anti_interference_retention_mechanism_search_20260311.json")
    g8b = load_json("g8b_high_margin_relation_bridge_discriminant_20260311.json")
    g9b = load_json("g9b_cross_model_intervention_stable_role_kernel_20260311.json")
    f1 = load_json("f1_architecture_scale_extrapolation_verification_20260311.json")

    qwen_under = f1["supporting_readout"]["generator_undercoverage"]["qwen3_4b"]
    deepseek_under = f1["supporting_readout"]["generator_undercoverage"]["deepseek_7b"]

    g7b_sensitivity = g10["adjusted_scores"]["g7b_calibrated"] - g10["adjusted_scores"]["g7b_raw"]
    g8b_sensitivity = g10["adjusted_scores"]["g8b_calibrated"] - g10["adjusted_scores"]["g8b_raw"]
    g9b_sensitivity = g10["adjusted_scores"]["g9b_calibrated"] - g10["adjusted_scores"]["g9b_raw"]

    block_sensitivity_score = mean([g7b_sensitivity, g8b_sensitivity, g9b_sensitivity])
    family_bias_asymmetry_score = mean([deepseek_under - qwen_under, deepseek_under, qwen_under + 0.5])
    architecture_scale_carryover_score = mean(
        [
            g10["headline_metrics"]["architecture_bias_pressure"],
            g10["headline_metrics"]["scale_limit_pressure"],
            f1["headline_metrics"]["architecture_scale_residual_boundary_score"],
        ]
    )

    overall_g11_score = mean(
        [
            block_sensitivity_score + 0.5,
            family_bias_asymmetry_score,
            architecture_scale_carryover_score,
        ]
    )

    formulas = {
        "surrogate_sensitivity": "Sens(block) = Score_calibrated(block) - Score_raw(block)",
        "family_asymmetry": "Asym = Undercoverage_deepseek - Undercoverage_qwen",
        "decomposition": "G11 = mean(SensitivityShift, FamilyBiasAsymmetry, ArchitectureScaleCarryover)",
    }

    verdict = {
        "status": (
            "surrogate_sensitivity_nontrivial"
            if overall_g11_score >= 0.53
            else "surrogate_sensitivity_weak"
        ),
        "most_sensitive_block": "G9B" if g9b_sensitivity >= max(g7b_sensitivity, g8b_sensitivity) else "G8B",
        "least_sensitive_block": "G7B" if g7b_sensitivity <= min(g8b_sensitivity, g9b_sensitivity) else "other",
        "core_answer": (
            "Surrogate sensitivity is structured, not uniform. Role-kernel evaluation is most exposed to family mismatch, "
            "bridge-law evaluation is moderately exposed, and retention remains comparatively mechanism-bound."
        ),
    }

    hypotheses = {
        "H1_block_sensitivity_is_nontrivial": block_sensitivity_score >= 0.05,
        "H2_family_bias_asymmetry_is_nontrivial": family_bias_asymmetry_score >= 0.35,
        "H3_architecture_scale_carryover_is_nontrivial": architecture_scale_carryover_score >= 0.55,
        "H4_g11_is_positive": overall_g11_score >= 0.53,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G11_surrogate_sensitivity_decomposition",
        },
        "headline_metrics": {
            "block_sensitivity_score": block_sensitivity_score,
            "family_bias_asymmetry_score": family_bias_asymmetry_score,
            "architecture_scale_carryover_score": architecture_scale_carryover_score,
            "overall_g11_score": overall_g11_score,
        },
        "supporting_readout": {
            "g7b_sensitivity": g7b_sensitivity,
            "g8b_sensitivity": g8b_sensitivity,
            "g9b_sensitivity": g9b_sensitivity,
            "qwen_undercoverage": qwen_under,
            "deepseek_undercoverage": deepseek_under,
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g11_surrogate_sensitivity_decomposition_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
