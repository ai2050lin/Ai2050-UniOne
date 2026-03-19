from __future__ import annotations

from stage56_generation_gate_internal_map import (
    aggregate_axis_blocks,
    aggregate_axis_rows,
    build_layer_profile,
    dominant_head_label,
    dominant_layer_label,
)


def test_build_layer_profile_averages_selected_gate_indices_by_layer() -> None:
    gate_by_layer = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    profile = build_layer_profile(gate_by_layer, [0, 3, 5], d_ff=4)
    assert profile == [2.5, 6.0]


def test_dominant_labels_pick_largest_magnitude_coordinate() -> None:
    assert dominant_layer_label([0.1, -0.4, 0.2]) == "layer_1"
    assert dominant_head_label([[0.0, 0.1], [-0.5, 0.2]]) == "layer_1_head_0"


def test_aggregate_axis_rows_and_blocks_keep_profiles_consistent() -> None:
    rows = [
        {
            "prototype_gate_delta": 0.1,
            "instance_gate_delta": -0.1,
            "strong_gate_delta": 0.2,
            "mixed_gate_delta": 0.3,
            "bridge_gate_delta": 0.1,
            "hidden_shift_profile": [0.2, 0.5],
            "mlp_layer_delta_profile": [0.1, 0.4],
            "attention_head_delta_profile": [[0.0, 0.2], [0.3, 0.1]],
        },
        {
            "prototype_gate_delta": 0.3,
            "instance_gate_delta": 0.1,
            "strong_gate_delta": 0.4,
            "mixed_gate_delta": 0.6,
            "bridge_gate_delta": 0.2,
            "hidden_shift_profile": [0.1, 0.6],
            "mlp_layer_delta_profile": [0.3, 0.2],
            "attention_head_delta_profile": [[0.0, 0.1], [0.2, 0.5]],
        },
    ]
    block = aggregate_axis_rows(rows)
    assert block["variant_count"] == 2
    assert abs(block["mean_bridge_gate_delta"] - 0.15) < 1e-9
    assert block["dominant_hidden_layer"] == "layer_1"

    merged = aggregate_axis_blocks([block, block])
    assert merged["variant_count"] == 4
    assert abs(merged["mean_bridge_gate_delta"] - 0.15) < 1e-9
    assert merged["dominant_mlp_layer"] == "layer_1"
