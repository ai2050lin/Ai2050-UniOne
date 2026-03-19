from stage56_natural_generation_strong_decoupling import build_summary, split_windows


def test_split_windows_separates_prompt_and_generated_regions() -> None:
    out = split_windows([1.0, 1.0, 4.0, 4.0], generated_token_count=2)
    assert out["prompt_sum"] == 2.0
    assert out["generated_sum"] == 8.0
    assert round(out["generated_share"], 4) == 0.8


def test_build_summary_marks_generated_dominant_component() -> None:
    natural_rows = [
        {
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "axis": "logic",
            "generated_token_count": 2,
            "tail_position_labels": ["tail_pos_-4", "tail_pos_-3", "tail_pos_-2", "tail_pos_-1"],
            "mean_hidden_token_profile": [1.0, 1.0, 4.0, 4.0],
            "mean_mlp_token_profile": [1.0, 1.0, 5.0, 5.0],
        },
        {
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "plum",
            "axis": "logic",
            "generated_token_count": 2,
            "tail_position_labels": ["tail_pos_-4", "tail_pos_-3", "tail_pos_-2", "tail_pos_-1"],
            "mean_hidden_token_profile": [1.0, 1.0, 6.0, 6.0],
            "mean_mlp_token_profile": [1.0, 1.0, 7.0, 7.0],
        },
    ]
    component_rows = [
        {
            "component_label": "logic_prototype",
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "weight": 0.5,
            "union_synergy_joint": 0.2,
            "union_joint_adv": 0.3,
        },
        {
            "component_label": "logic_prototype",
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "plum",
            "weight": 0.5,
            "union_synergy_joint": 0.4,
            "union_joint_adv": 0.5,
        },
    ]
    summary = build_summary(natural_rows, component_rows)
    assert summary["case_count"] == 2
    assert summary["per_component"]["logic_prototype"]["signal_origin"] in {"generated_dominant", "mixed"}
