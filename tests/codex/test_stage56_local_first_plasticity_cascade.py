try:
    from tests.codex.stage56_local_first_plasticity_cascade import build_local_first_cascade_summary
except ModuleNotFoundError:
    from stage56_local_first_plasticity_cascade import build_local_first_cascade_summary


def test_local_first_cascade_summary_has_expected_shape() -> None:
    summary = build_local_first_cascade_summary()
    hm = summary["headline_metrics"]
    assert hm["frontier_peak"] > hm["boundary_peak"] > hm["atlas_peak"]
    assert hm["local_to_boundary_ratio"] > 1.0
    assert hm["boundary_to_atlas_ratio"] > 1.0
    assert hm["same_ordering"] is True
    assert "local_patch_update" in summary["cascade_equation"]
