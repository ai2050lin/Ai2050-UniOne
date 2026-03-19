from stage56_density_frontier_closure_link import decompose_axis_fields, pearson


def test_decompose_axis_fields_splits_supportive_and_fragile_parts():
    row = {
        "union_joint_adv": 0.2,
        "union_synergy_joint": 0.1,
        "strict_positive_synergy": True,
        "axes": {
            "logic": {
                "bridge_field_proxy": 0.5,
                "conflict_field_proxy": 0.4,
                "mismatch_field_proxy": 0.3,
            }
        },
    }
    out = decompose_axis_fields(row, "logic")
    assert out["stable_bridge"] == 0.5
    assert out["fragile_bridge"] == 0.0
    assert out["constraint_conflict"] == 0.4
    assert out["mismatch_exposure"] == 0.3


def test_pearson_matches_positive_trend():
    xs = [1.0, 2.0, 3.0]
    ys = [2.0, 4.0, 6.0]
    assert round(pearson(xs, ys), 6) == 1.0
