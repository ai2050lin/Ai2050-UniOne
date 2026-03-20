try:
    from tests.codex.stage56_native_stage_detector import build_native_stage_detector_summary
except ModuleNotFoundError:
    from stage56_native_stage_detector import build_native_stage_detector_summary


def test_native_stage_detector_preserves_order() -> None:
    summary = build_native_stage_detector_summary()
    assert summary["case_count"] >= 4
    assert summary["ordered_ratio"] == 1.0

