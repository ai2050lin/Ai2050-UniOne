from __future__ import annotations

from stage56_static_term_estimator import build_static_estimates


def test_build_static_estimates_produces_normalized_static_terms() -> None:
    frontier_summary = {
        "density_frontier": {
            "broad_support_base": 0.19,
            "long_separation_frontier": 0.23,
            "strongest_positive_frontier": {"corr": 0.31},
            "strongest_negative_frontier": {"corr": -0.39},
        },
        "closure": {
            "pair_positive_ratio": 0.19,
        },
    }
    unified_summary = {
        "normalized_coefficients": {
            "atlas_static": 0.24,
            "offset_static": 0.16,
        }
    }

    out = build_static_estimates(frontier_summary, unified_summary)
    normalized_static = out["normalized_static"]
    assert out["record_type"] == "stage56_static_term_estimator_summary"
    assert abs(sum(normalized_static.values()) - 1.0) < 1e-9
    assert normalized_static["atlas_static_hat"] > 0.0
    assert normalized_static["offset_static_hat"] > 0.0
