from stage56_pair_density_field_closure import build_summary, longest_positive_band


def test_longest_positive_band_prefers_longer_positive_segment() -> None:
    rows = [
        {"mass_ratio": 0.1, "corr": 0.1},
        {"mass_ratio": 0.2, "corr": 0.3},
        {"mass_ratio": 0.3, "corr": 0.31},
        {"mass_ratio": 0.4, "corr": 0.0},
        {"mass_ratio": 0.5, "corr": 0.25},
    ]
    band = longest_positive_band(rows, threshold=0.2)
    assert band["length"] == 2
    assert band["start_mass_ratio"] == 0.2
    assert band["end_mass_ratio"] == 0.3


def test_build_summary_collects_mass_point_correlations() -> None:
    joined_rows = [
        {
            "strict_positive_synergy": True,
            "union_joint_adv": 0.5,
            "union_synergy_joint": 0.4,
            "axes": {
                "logic": {"pair_compaction_curve": {"10": 0.2, "25": 0.3}, "pair_coverage_curve": {"10": 0.1, "25": 0.2}},
                "style": {"pair_compaction_curve": {"10": 0.1, "25": 0.2}, "pair_coverage_curve": {"10": 0.2, "25": 0.3}},
                "syntax": {"pair_compaction_curve": {"10": 0.4, "25": 0.5}, "pair_coverage_curve": {"10": 0.1, "25": 0.1}},
            },
        },
        {
            "strict_positive_synergy": False,
            "union_joint_adv": -0.2,
            "union_synergy_joint": -0.3,
            "axes": {
                "logic": {"pair_compaction_curve": {"10": 0.6, "25": 0.7}, "pair_coverage_curve": {"10": 0.8, "25": 0.9}},
                "style": {"pair_compaction_curve": {"10": 0.5, "25": 0.6}, "pair_coverage_curve": {"10": 0.6, "25": 0.7}},
                "syntax": {"pair_compaction_curve": {"10": 0.2, "25": 0.2}, "pair_coverage_curve": {"10": 0.9, "25": 0.9}},
            },
        },
    ]
    summary = build_summary(joined_rows, [0.10, 0.25])
    assert summary["joined_pair_count"] == 2
    assert summary["mass_ratio_count"] == 2
    assert summary["top_abs_mass_point_correlations"]
