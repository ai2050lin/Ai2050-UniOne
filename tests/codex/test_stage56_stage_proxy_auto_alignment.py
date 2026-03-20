try:
    from tests.codex.stage56_stage_proxy_auto_alignment import build_stage_proxy_auto_alignment_summary
except ModuleNotFoundError:
    from stage56_stage_proxy_auto_alignment import build_stage_proxy_auto_alignment_summary


def test_stage_proxy_auto_alignment_ordered_ratio_is_full() -> None:
    summary = build_stage_proxy_auto_alignment_summary()
    assert summary["case_count"] >= 4
    assert summary["ordered_ratio"] == 1.0

