from stage56_pair_density_tensor_field import build_axis_tensor, build_summary


def test_build_axis_tensor_keeps_role_channel_structure() -> None:
    proto = {
        "compaction_curve": {"10": 0.2, "20": 0.3},
        "coverage_curve": {"10": 0.5, "20": 0.6},
        "delta_l2": 2.0,
        "delta_mean_abs": 0.4,
    }
    inst = {
        "compaction_curve": {"10": 0.1, "20": 0.2},
        "coverage_curve": {"10": 0.4, "20": 0.7},
        "delta_l2": 4.0,
        "delta_mean_abs": 0.8,
    }
    out = build_axis_tensor(proto, inst, [0.10, 0.20])
    assert out["tensor_shape"] == [2, 2, 2]
    assert out["prototype_compaction_curve"] == [0.2, 0.3]
    assert out["instance_coverage_curve"] == [0.4, 0.7]
    assert out["pair_delta_l2"] == 3.0


def test_build_summary_collects_tensor_correlations() -> None:
    rows = [
        {
            "strict_positive_synergy": True,
            "union_joint_adv": 0.4,
            "union_synergy_joint": 0.3,
            "axes": {
                "logic": {
                    "role_asymmetry_compaction_l1": 0.1,
                    "role_asymmetry_coverage_l1": 0.2,
                    "channel_alignment_proto": 0.8,
                    "channel_alignment_instance": 0.7,
                    "role_alignment_compaction": 0.9,
                    "role_alignment_coverage": 0.85,
                    "pair_compaction_early_mean": 0.2,
                    "pair_compaction_middle_mean": 0.3,
                    "pair_compaction_late_mean": 0.4,
                    "pair_coverage_early_mean": 0.2,
                    "pair_coverage_middle_mean": 0.25,
                    "pair_coverage_late_mean": 0.3,
                    "pair_delta_l2": 2.0,
                    "pair_delta_mean_abs": 0.4,
                },
                "style": {
                    "role_asymmetry_compaction_l1": 0.2,
                    "role_asymmetry_coverage_l1": 0.3,
                    "channel_alignment_proto": 0.4,
                    "channel_alignment_instance": 0.5,
                    "role_alignment_compaction": 0.3,
                    "role_alignment_coverage": 0.2,
                    "pair_compaction_early_mean": 0.3,
                    "pair_compaction_middle_mean": 0.4,
                    "pair_compaction_late_mean": 0.5,
                    "pair_coverage_early_mean": 0.4,
                    "pair_coverage_middle_mean": 0.5,
                    "pair_coverage_late_mean": 0.6,
                    "pair_delta_l2": 1.0,
                    "pair_delta_mean_abs": 0.2,
                },
                "syntax": {
                    "role_asymmetry_compaction_l1": 0.05,
                    "role_asymmetry_coverage_l1": 0.1,
                    "channel_alignment_proto": 0.9,
                    "channel_alignment_instance": 0.9,
                    "role_alignment_compaction": 0.95,
                    "role_alignment_coverage": 0.95,
                    "pair_compaction_early_mean": 0.5,
                    "pair_compaction_middle_mean": 0.6,
                    "pair_compaction_late_mean": 0.7,
                    "pair_coverage_early_mean": 0.5,
                    "pair_coverage_middle_mean": 0.6,
                    "pair_coverage_late_mean": 0.7,
                    "pair_delta_l2": 3.0,
                    "pair_delta_mean_abs": 0.5,
                },
            },
        },
        {
            "strict_positive_synergy": False,
            "union_joint_adv": -0.2,
            "union_synergy_joint": -0.3,
            "axes": {
                "logic": {
                    "role_asymmetry_compaction_l1": 0.4,
                    "role_asymmetry_coverage_l1": 0.5,
                    "channel_alignment_proto": 0.2,
                    "channel_alignment_instance": 0.1,
                    "role_alignment_compaction": 0.1,
                    "role_alignment_coverage": 0.2,
                    "pair_compaction_early_mean": 0.8,
                    "pair_compaction_middle_mean": 0.7,
                    "pair_compaction_late_mean": 0.6,
                    "pair_coverage_early_mean": 0.8,
                    "pair_coverage_middle_mean": 0.7,
                    "pair_coverage_late_mean": 0.6,
                    "pair_delta_l2": 5.0,
                    "pair_delta_mean_abs": 0.9,
                },
                "style": {
                    "role_asymmetry_compaction_l1": 0.3,
                    "role_asymmetry_coverage_l1": 0.4,
                    "channel_alignment_proto": 0.3,
                    "channel_alignment_instance": 0.2,
                    "role_alignment_compaction": 0.2,
                    "role_alignment_coverage": 0.1,
                    "pair_compaction_early_mean": 0.6,
                    "pair_compaction_middle_mean": 0.5,
                    "pair_compaction_late_mean": 0.4,
                    "pair_coverage_early_mean": 0.7,
                    "pair_coverage_middle_mean": 0.6,
                    "pair_coverage_late_mean": 0.5,
                    "pair_delta_l2": 4.0,
                    "pair_delta_mean_abs": 0.8,
                },
                "syntax": {
                    "role_asymmetry_compaction_l1": 0.3,
                    "role_asymmetry_coverage_l1": 0.4,
                    "channel_alignment_proto": 0.1,
                    "channel_alignment_instance": 0.1,
                    "role_alignment_compaction": 0.1,
                    "role_alignment_coverage": 0.1,
                    "pair_compaction_early_mean": 0.1,
                    "pair_compaction_middle_mean": 0.2,
                    "pair_compaction_late_mean": 0.3,
                    "pair_coverage_early_mean": 0.1,
                    "pair_coverage_middle_mean": 0.2,
                    "pair_coverage_late_mean": 0.3,
                    "pair_delta_l2": 6.0,
                    "pair_delta_mean_abs": 1.0,
                },
            },
        },
    ]
    summary = build_summary(rows)
    assert summary["joined_pair_count"] == 2
    assert summary["axis_row_count"] == 6
    assert summary["top_abs_correlations"]
