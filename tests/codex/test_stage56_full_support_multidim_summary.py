from stage56_full_support_multidim_summary import derive_system_claims, summarize_dimension


def test_summarize_dimension_computes_compaction():
    row = summarize_dimension(
        {
            "selection_mode": "all_effective",
            "selected_neuron_count": 80,
            "mass10_neuron_count": 2,
            "mass25_neuron_count": 8,
            "mass50_neuron_count": 4,
            "mass80_neuron_count": 10,
            "mass95_neuron_count": 20,
            "mass10_layer_coverage": {"covered_layer_ratio": 0.25},
            "mass25_layer_coverage": {"covered_layer_ratio": 0.4},
            "mass50_layer_coverage": {"covered_layer_ratio": 0.5},
            "mass80_layer_coverage": {"covered_layer_ratio": 0.8},
            "mass95_layer_coverage": {"covered_layer_ratio": 1.0},
            "pair_delta_cosine_mean": 0.2,
            "specific_selected_count": 7,
            "specific_selected_ratio": 0.07,
        },
        total_neurons=100,
    )

    assert row["selected_neuron_ratio"] == 0.8
    assert row["mass10_compaction_ratio"] == 0.025
    assert row["mass25_compaction_ratio"] == 0.1
    assert row["mass50_compaction_ratio"] == 0.05
    assert row["mass80_compaction_ratio"] == 0.125


def test_derive_system_claims_prefers_broad_support_narrow_core():
    per_dim = {
        "style": {"selected_neuron_ratio": 0.95, "mass10_compaction_ratio": 0.03, "mass25_compaction_ratio": 0.05, "mass50_compaction_ratio": 0.09},
        "logic": {"selected_neuron_ratio": 0.92, "mass10_compaction_ratio": 0.04, "mass25_compaction_ratio": 0.06, "mass50_compaction_ratio": 0.10},
        "syntax": {"selected_neuron_ratio": 0.90, "mass10_compaction_ratio": 0.05, "mass25_compaction_ratio": 0.07, "mass50_compaction_ratio": 0.11},
    }
    cross = {
        "style__logic": {
            "effective_support_jaccard": 0.88,
            "mass10_jaccard": 0.05,
            "mass25_jaccard": 0.09,
            "mass50_jaccard": 0.12,
            "mass80_jaccard": 0.20,
            "layer_profile_corr": 0.70,
        },
        "style__syntax": {
            "effective_support_jaccard": 0.86,
            "mass10_jaccard": 0.04,
            "mass25_jaccard": 0.08,
            "mass50_jaccard": 0.10,
            "mass80_jaccard": 0.19,
            "layer_profile_corr": 0.75,
        },
        "logic__syntax": {
            "effective_support_jaccard": 0.83,
            "mass10_jaccard": 0.03,
            "mass25_jaccard": 0.07,
            "mass50_jaccard": 0.08,
            "mass80_jaccard": 0.17,
            "layer_profile_corr": 0.72,
        },
    }

    claims = derive_system_claims(per_dim, cross)

    assert claims["support_shape"] == "广支撑-窄核心"
    assert claims["mean_mass25_jaccard"] < claims["mean_effective_support_jaccard"]
