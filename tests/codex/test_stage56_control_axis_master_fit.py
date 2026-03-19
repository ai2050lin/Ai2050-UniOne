from __future__ import annotations

from stage56_control_axis_master_fit import build_control_axis_fit


def test_build_control_axis_fit_keeps_signed_axis_direction() -> None:
    master_fit_summary = {
        "fitted_weights": {
            "atlas_static": 0.1,
            "offset_static": 0.1,
            "frontier_dynamic": 0.2,
            "subfield_dynamic": 0.2,
            "window_closure": 0.2,
            "closure_boundary": 0.2,
        }
    }
    pair_link_summary = {
        "axis_target_stats": {
            "style": {
                "prototype_field_proxy": {
                    "targets": {"union_synergy_joint": {"pearson_corr": -0.24}}
                }
            },
            "logic": {
                "prototype_field_proxy": {
                    "targets": {"union_synergy_joint": {"pearson_corr": 0.25}}
                },
                "bridge_field_proxy": {
                    "targets": {"union_synergy_joint": {"pearson_corr": -0.20}}
                },
            },
            "syntax": {
                "conflict_field_proxy": {
                    "targets": {"union_synergy_joint": {"pearson_corr": 0.10}}
                },
                "bridge_field_proxy": {
                    "targets": {"union_synergy_joint": {"pearson_corr": 0.08}}
                },
            },
        }
    }

    out = build_control_axis_fit(master_fit_summary, pair_link_summary)
    normalized_control = out["normalized_control"]
    assert out["record_type"] == "stage56_control_axis_master_fit_summary"
    assert normalized_control["style_control"] < 0.0
    assert normalized_control["syntax_control"] > 0.0
    assert "Style_control" in out["equation_text"]
