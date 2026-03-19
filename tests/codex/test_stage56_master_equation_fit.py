from __future__ import annotations

from stage56_master_equation_fit import build_fit


def test_build_fit_returns_normalized_weights() -> None:
    law_summary = {
        "laws": {
            "broad_support_base": 0.19,
            "long_separation_frontier": 0.23,
            "mid_syntax_filter": 0.30,
        }
    }
    frontier_summary = {
        "density_frontier": {
            "strongest_positive_frontier": {"corr": 0.31},
            "strongest_negative_frontier": {"corr": -0.39},
        },
        "internal_subfield": {
            "components": [
                {"best_positive_corr_to_synergy": 0.18, "best_negative_corr_to_synergy": -0.22},
                {"best_positive_corr_to_synergy": 0.17, "best_negative_corr_to_synergy": -0.10},
            ]
        },
        "token_window": {
            "components": [
                {"mean_union_synergy_joint": -0.06},
                {"mean_union_synergy_joint": 0.04},
            ]
        },
        "closure": {
            "pair_positive_ratio": 0.19,
            "strongest_positive_field_to_synergy": {"corr": 0.25},
            "strongest_negative_field_to_synergy": {"corr": -0.24},
        },
    }
    unified_summary = {
        "normalized_coefficients": {
            "atlas_static": 0.20,
            "offset_static": 0.15,
        }
    }

    out = build_fit(law_summary, frontier_summary, unified_summary)
    weights = out["fitted_weights"]
    assert out["record_type"] == "stage56_master_equation_fit_summary"
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["window_closure"] > 0.0
