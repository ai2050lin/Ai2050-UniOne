from __future__ import annotations

from stage56_simple_generator_laws import build_laws


def test_build_laws_produces_five_core_terms() -> None:
    summary = {
        "per_component": {
            "logic_prototype": {
                "feature_stats": {
                    "preferred_density": {"mean_value": 0.2, "targets": {"strict_positive_synergy": {"pearson_corr": 0.1}}},
                    "hidden_layer_center": {"targets": {"strict_positive_synergy": {"pearson_corr": 0.3}}},
                    "mlp_layer_center": {"targets": {"strict_positive_synergy": {"pearson_corr": 0.2}}},
                }
            },
            "logic_fragile_bridge": {
                "feature_stats": {
                    "preferred_density": {"mean_value": 0.25, "targets": {"strict_positive_synergy": {"pearson_corr": -0.2}}},
                    "hidden_window_center": {"targets": {"strict_positive_synergy": {"pearson_corr": -0.1}}},
                    "mlp_window_center": {"targets": {"strict_positive_synergy": {"pearson_corr": -0.2}}},
                    "mlp_generated_share": {"targets": {"strict_positive_synergy": {"pearson_corr": 0.15}}},
                }
            },
            "syntax_constraint_conflict": {
                "feature_stats": {
                    "preferred_density": {"mean_value": 0.15, "targets": {"strict_positive_synergy": {"pearson_corr": 0.4}}},
                    "complete_generated_energy": {"targets": {"strict_positive_synergy": {"pearson_corr": 0.3}}},
                }
            },
        }
    }
    out = build_laws(summary)
    assert set(out["laws"].keys()) == {
        "broad_support_base",
        "long_separation_frontier",
        "late_skeleton_shift",
        "mid_syntax_filter",
        "late_window_closure",
    }
    assert out["closure_score"] > 0.0
