from stage56_multimodel_full_support_compare import derive_model_private_signals, derive_shared_claims


def test_derive_shared_claims_prefers_broad_support_narrow_core():
    rows = [
        {
            "label": "DeepSeek-7B",
            "support_shape": "广支撑-窄核心",
            "narrow_core_dimension": "syntax",
            "stable_axis_dimension": "logic",
            "broad_reconfiguration_dimension": "style",
            "mean_effective_support_jaccard": 0.99,
            "mean_mass10_jaccard": 0.08,
            "mean_mass25_jaccard": 0.13,
            "mean_mass50_jaccard": 0.30,
            "mean_mass10_compaction": 0.04,
            "mean_mass25_compaction": 0.11,
            "mean_layer_profile_corr": 0.72,
        },
        {
            "label": "GLM-4-9B",
            "support_shape": "广支撑-窄核心",
            "narrow_core_dimension": "syntax",
            "stable_axis_dimension": "logic",
            "broad_reconfiguration_dimension": "style",
            "mean_effective_support_jaccard": 0.98,
            "mean_mass10_jaccard": 0.10,
            "mean_mass25_jaccard": 0.15,
            "mean_mass50_jaccard": 0.33,
            "mean_mass10_compaction": 0.05,
            "mean_mass25_compaction": 0.12,
            "mean_layer_profile_corr": 0.68,
        },
        {
            "label": "Qwen3-4B",
            "support_shape": "广支撑-窄核心",
            "narrow_core_dimension": "syntax",
            "stable_axis_dimension": "logic",
            "broad_reconfiguration_dimension": "style",
            "mean_effective_support_jaccard": 0.97,
            "mean_mass10_jaccard": 0.12,
            "mean_mass25_jaccard": 0.16,
            "mean_mass50_jaccard": 0.35,
            "mean_mass10_compaction": 0.06,
            "mean_mass25_compaction": 0.14,
            "mean_layer_profile_corr": 0.66,
        },
    ]

    claims = derive_shared_claims(rows)

    assert claims["shared_support_shape"] == "广支撑-窄核心"
    assert claims["shared_narrow_core_dimension"] == "syntax"
    assert claims["shared_stable_axis_dimension"] == "logic"
    assert claims["shared_broad_reconfiguration_dimension"] == "style"
    assert claims["mean_mass25_jaccard"] < claims["mean_effective_support_jaccard"]


def test_derive_model_private_signals_is_centered():
    rows = [
        {
            "label": "A",
            "mean_mass10_compaction": 0.03,
            "mean_layer_profile_corr": 0.80,
            "stable_axis_value": 0.50,
            "broad_reconfiguration_value": 0.70,
        },
        {
            "label": "B",
            "mean_mass10_compaction": 0.05,
            "mean_layer_profile_corr": 0.70,
            "stable_axis_value": 0.40,
            "broad_reconfiguration_value": 0.60,
        },
    ]

    deltas = derive_model_private_signals(rows)

    assert deltas["A"]["mass10_compaction_delta_vs_mean"] == -0.01
    assert deltas["B"]["layer_profile_corr_delta_vs_mean"] == -0.05
    assert deltas["A"]["stable_axis_strength_delta_vs_mean"] == 0.05
