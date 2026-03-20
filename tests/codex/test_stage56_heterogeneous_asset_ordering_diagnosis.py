try:
    from tests.codex.stage56_heterogeneous_asset_ordering_diagnosis import build_ordering_diagnosis_summary
except ModuleNotFoundError:
    from stage56_heterogeneous_asset_ordering_diagnosis import build_ordering_diagnosis_summary


def test_ordering_diagnosis_has_gap() -> None:
    summary = build_ordering_diagnosis_summary()
    assert summary["refined_order_ratio"] >= summary["coarse_order_ratio"]
