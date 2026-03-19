from stage56_density_frontier_curve import (
    auc_gap,
    first_mass_ratio_for_coverage,
    first_mass_ratio_for_jaccard,
    knee_mass_ratio,
)


def test_auc_gap_prefers_sharper_curve():
    sharp = [
        {"mass_ratio": 0.1, "compaction_ratio": 0.02},
        {"mass_ratio": 0.2, "compaction_ratio": 0.05},
        {"mass_ratio": 0.3, "compaction_ratio": 0.09},
    ]
    flat = [
        {"mass_ratio": 0.1, "compaction_ratio": 0.08},
        {"mass_ratio": 0.2, "compaction_ratio": 0.16},
        {"mass_ratio": 0.3, "compaction_ratio": 0.24},
    ]
    assert auc_gap(sharp) > auc_gap(flat)


def test_knee_mass_ratio_finds_mid_curve_turn():
    curve = [
        {"mass_ratio": 0.01, "compaction_ratio": 0.001},
        {"mass_ratio": 0.05, "compaction_ratio": 0.010},
        {"mass_ratio": 0.10, "compaction_ratio": 0.030},
        {"mass_ratio": 0.20, "compaction_ratio": 0.120},
        {"mass_ratio": 0.30, "compaction_ratio": 0.210},
        {"mass_ratio": 0.40, "compaction_ratio": 0.290},
        {"mass_ratio": 0.50, "compaction_ratio": 0.360},
    ]
    assert knee_mass_ratio(curve) == 0.2


def test_first_mass_ratio_helpers():
    coverage_curve = [
        {"mass_ratio": 0.01, "layer_coverage_ratio": 0.6},
        {"mass_ratio": 0.05, "layer_coverage_ratio": 0.9},
        {"mass_ratio": 0.10, "layer_coverage_ratio": 1.0},
    ]
    jaccard_curve = [
        {"mass_ratio": 0.01, "jaccard": 0.02},
        {"mass_ratio": 0.10, "jaccard": 0.20},
        {"mass_ratio": 0.30, "jaccard": 0.55},
    ]
    assert first_mass_ratio_for_coverage(coverage_curve, 1.0) == 0.10
    assert first_mass_ratio_for_jaccard(jaccard_curve, 0.5) == 0.30
