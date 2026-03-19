from stage56_natural_pair_frontier_closure_link import (
    build_summary,
    join_pair_rows,
    load_pairs_manifest,
)


def test_load_pairs_manifest_reads_term_and_category(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(
        """{
  "pairs": {
    "logic": [
      {"id": "logic_fruit_0000_apple", "term": "apple", "category": "fruit"}
    ]
  }
}""",
        encoding="utf-8",
    )
    out = load_pairs_manifest(path)
    assert out[("logic", "logic_fruit_0000_apple")]["term"] == "apple"
    assert out[("logic", "logic_fruit_0000_apple")]["category"] == "fruit"


def test_join_pair_rows_matches_prototype_and_instance_terms(tmp_path):
    joined_rows_path = tmp_path / "joined_rows.jsonl"
    joined_rows_path.write_text(
        '{"model_id":"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","category":"fruit","prototype_term":"apple","instance_term":"pear","strict_positive_synergy":true,"union_joint_adv":0.2,"union_synergy_joint":0.1}\n',
        encoding="utf-8",
    )
    term_metrics = {
        ("DeepSeek-7B", "logic", "fruit", "apple"): {
            "delta_l2": 3.0,
            "delta_mean_abs": 1.0,
            "delta_l2_topness": 1.0,
            "delta_mean_abs_topness": 0.5,
            "delta_l2_zscore": 0.2,
            "delta_mean_abs_zscore": 0.1,
        },
        ("DeepSeek-7B", "logic", "fruit", "pear"): {
            "delta_l2": 1.0,
            "delta_mean_abs": 0.4,
            "delta_l2_topness": 0.0,
            "delta_mean_abs_topness": 0.2,
            "delta_l2_zscore": -0.2,
            "delta_mean_abs_zscore": -0.1,
        },
        ("DeepSeek-7B", "style", "fruit", "apple"): {
            "delta_l2": 2.0,
            "delta_mean_abs": 0.9,
            "delta_l2_topness": 0.9,
            "delta_mean_abs_topness": 0.4,
            "delta_l2_zscore": 0.3,
            "delta_mean_abs_zscore": 0.2,
        },
        ("DeepSeek-7B", "style", "fruit", "pear"): {
            "delta_l2": 2.5,
            "delta_mean_abs": 0.7,
            "delta_l2_topness": 0.6,
            "delta_mean_abs_topness": 0.3,
            "delta_l2_zscore": 0.1,
            "delta_mean_abs_zscore": 0.0,
        },
        ("DeepSeek-7B", "syntax", "fruit", "apple"): {
            "delta_l2": 4.0,
            "delta_mean_abs": 1.1,
            "delta_l2_topness": 1.0,
            "delta_mean_abs_topness": 1.0,
            "delta_l2_zscore": 0.6,
            "delta_mean_abs_zscore": 0.5,
        },
        ("DeepSeek-7B", "syntax", "fruit", "pear"): {
            "delta_l2": 3.0,
            "delta_mean_abs": 0.8,
            "delta_l2_topness": 0.5,
            "delta_mean_abs_topness": 0.4,
            "delta_l2_zscore": 0.2,
            "delta_mean_abs_zscore": 0.1,
        },
    }
    out = join_pair_rows(joined_rows_path, term_metrics)
    assert len(out) == 1
    assert out[0]["axes"]["logic"]["pair_mean_delta_l2"] == 2.0
    assert out[0]["axes"]["logic"]["pair_gap_delta_mean_abs"] == 0.6


def test_build_summary_emits_top_correlations():
    rows = [
        {
            "model_label": "DeepSeek-7B",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "strict_positive_synergy": True,
            "union_joint_adv": 0.3,
            "union_synergy_joint": 0.2,
            "axes": {
                "logic": {
                    "prototype_delta_l2": 3.0,
                    "instance_delta_l2": 1.0,
                    "pair_mean_delta_l2": 2.0,
                    "pair_gap_delta_l2": 2.0,
                    "prototype_delta_mean_abs": 1.0,
                    "instance_delta_mean_abs": 0.4,
                    "pair_mean_delta_mean_abs": 0.7,
                    "pair_gap_delta_mean_abs": 0.6,
                    "prototype_delta_l2_topness": 1.0,
                    "instance_delta_l2_topness": 0.0,
                    "pair_mean_delta_l2_topness": 0.5,
                    "prototype_delta_mean_abs_topness": 0.8,
                    "instance_delta_mean_abs_topness": 0.2,
                    "pair_mean_delta_mean_abs_topness": 0.5,
                    "prototype_delta_l2_zscore": 0.4,
                    "instance_delta_l2_zscore": -0.4,
                    "pair_mean_delta_l2_zscore": 0.0,
                    "prototype_delta_mean_abs_zscore": 0.3,
                    "instance_delta_mean_abs_zscore": -0.3,
                    "pair_mean_delta_mean_abs_zscore": 0.0,
                },
                "style": {
                    "prototype_delta_l2": 1.0,
                    "instance_delta_l2": 1.2,
                    "pair_mean_delta_l2": 1.1,
                    "pair_gap_delta_l2": 0.2,
                    "prototype_delta_mean_abs": 0.5,
                    "instance_delta_mean_abs": 0.6,
                    "pair_mean_delta_mean_abs": 0.55,
                    "pair_gap_delta_mean_abs": 0.1,
                    "prototype_delta_l2_topness": 0.3,
                    "instance_delta_l2_topness": 0.4,
                    "pair_mean_delta_l2_topness": 0.35,
                    "prototype_delta_mean_abs_topness": 0.2,
                    "instance_delta_mean_abs_topness": 0.4,
                    "pair_mean_delta_mean_abs_topness": 0.3,
                    "prototype_delta_l2_zscore": -0.2,
                    "instance_delta_l2_zscore": -0.1,
                    "pair_mean_delta_l2_zscore": -0.15,
                    "prototype_delta_mean_abs_zscore": -0.1,
                    "instance_delta_mean_abs_zscore": 0.0,
                    "pair_mean_delta_mean_abs_zscore": -0.05,
                },
                "syntax": {
                    "prototype_delta_l2": 2.5,
                    "instance_delta_l2": 2.0,
                    "pair_mean_delta_l2": 2.25,
                    "pair_gap_delta_l2": 0.5,
                    "prototype_delta_mean_abs": 0.9,
                    "instance_delta_mean_abs": 0.7,
                    "pair_mean_delta_mean_abs": 0.8,
                    "pair_gap_delta_mean_abs": 0.2,
                    "prototype_delta_l2_topness": 0.9,
                    "instance_delta_l2_topness": 0.7,
                    "pair_mean_delta_l2_topness": 0.8,
                    "prototype_delta_mean_abs_topness": 0.9,
                    "instance_delta_mean_abs_topness": 0.6,
                    "pair_mean_delta_mean_abs_topness": 0.75,
                    "prototype_delta_l2_zscore": 0.5,
                    "instance_delta_l2_zscore": 0.2,
                    "pair_mean_delta_l2_zscore": 0.35,
                    "prototype_delta_mean_abs_zscore": 0.4,
                    "instance_delta_mean_abs_zscore": 0.1,
                    "pair_mean_delta_mean_abs_zscore": 0.25,
                },
            },
        },
        {
            "model_label": "DeepSeek-7B",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "plum",
            "strict_positive_synergy": False,
            "union_joint_adv": -0.1,
            "union_synergy_joint": -0.2,
            "axes": {
                "logic": {
                    "prototype_delta_l2": 1.0,
                    "instance_delta_l2": 0.8,
                    "pair_mean_delta_l2": 0.9,
                    "pair_gap_delta_l2": 0.2,
                    "prototype_delta_mean_abs": 0.4,
                    "instance_delta_mean_abs": 0.3,
                    "pair_mean_delta_mean_abs": 0.35,
                    "pair_gap_delta_mean_abs": 0.1,
                    "prototype_delta_l2_topness": 0.2,
                    "instance_delta_l2_topness": 0.1,
                    "pair_mean_delta_l2_topness": 0.15,
                    "prototype_delta_mean_abs_topness": 0.2,
                    "instance_delta_mean_abs_topness": 0.1,
                    "pair_mean_delta_mean_abs_topness": 0.15,
                    "prototype_delta_l2_zscore": -0.3,
                    "instance_delta_l2_zscore": -0.4,
                    "pair_mean_delta_l2_zscore": -0.35,
                    "prototype_delta_mean_abs_zscore": -0.2,
                    "instance_delta_mean_abs_zscore": -0.3,
                    "pair_mean_delta_mean_abs_zscore": -0.25,
                },
                "style": {
                    "prototype_delta_l2": 1.1,
                    "instance_delta_l2": 1.0,
                    "pair_mean_delta_l2": 1.05,
                    "pair_gap_delta_l2": 0.1,
                    "prototype_delta_mean_abs": 0.45,
                    "instance_delta_mean_abs": 0.4,
                    "pair_mean_delta_mean_abs": 0.425,
                    "pair_gap_delta_mean_abs": 0.05,
                    "prototype_delta_l2_topness": 0.3,
                    "instance_delta_l2_topness": 0.25,
                    "pair_mean_delta_l2_topness": 0.275,
                    "prototype_delta_mean_abs_topness": 0.3,
                    "instance_delta_mean_abs_topness": 0.2,
                    "pair_mean_delta_mean_abs_topness": 0.25,
                    "prototype_delta_l2_zscore": -0.2,
                    "instance_delta_l2_zscore": -0.25,
                    "pair_mean_delta_l2_zscore": -0.225,
                    "prototype_delta_mean_abs_zscore": -0.15,
                    "instance_delta_mean_abs_zscore": -0.2,
                    "pair_mean_delta_mean_abs_zscore": -0.175,
                },
                "syntax": {
                    "prototype_delta_l2": 1.4,
                    "instance_delta_l2": 1.2,
                    "pair_mean_delta_l2": 1.3,
                    "pair_gap_delta_l2": 0.2,
                    "prototype_delta_mean_abs": 0.5,
                    "instance_delta_mean_abs": 0.45,
                    "pair_mean_delta_mean_abs": 0.475,
                    "pair_gap_delta_mean_abs": 0.05,
                    "prototype_delta_l2_topness": 0.4,
                    "instance_delta_l2_topness": 0.3,
                    "pair_mean_delta_l2_topness": 0.35,
                    "prototype_delta_mean_abs_topness": 0.35,
                    "instance_delta_mean_abs_topness": 0.25,
                    "pair_mean_delta_mean_abs_topness": 0.3,
                    "prototype_delta_l2_zscore": -0.1,
                    "instance_delta_l2_zscore": -0.2,
                    "pair_mean_delta_l2_zscore": -0.15,
                    "prototype_delta_mean_abs_zscore": -0.1,
                    "instance_delta_mean_abs_zscore": -0.15,
                    "pair_mean_delta_mean_abs_zscore": -0.125,
                },
            },
        },
    ]
    summary = build_summary(rows)
    assert summary["joined_pair_count"] == 2
    assert summary["axis_row_count"] == 6
    assert summary["top_abs_correlations"]
