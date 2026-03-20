try:
    from tests.codex.stage56_local_native_update_field import build_local_native_update_field_summary
except ModuleNotFoundError:
    from stage56_local_native_update_field import build_local_native_update_field_summary


def test_local_native_update_field_summary_shape() -> None:
    summary = build_local_native_update_field_summary()
    hm = summary["headline_metrics"]
    assert hm["patch_update_native"] > hm["boundary_response_native"] > hm["atlas_consolidation_native"]
    assert "local_field_equation" in summary

