try:
    from tests.codex.stage56_heterogeneous_asset_recanonicalization import build_recanonicalization_summary
except ModuleNotFoundError:
    from stage56_heterogeneous_asset_recanonicalization import build_recanonicalization_summary


def test_recanonicalization_improves_comparable_order_ratio() -> None:
    summary = build_recanonicalization_summary()
    assert summary["recanonicalized_comparable_ratio"] >= summary["coarse_order_ratio"]
    assert summary["comparable_case_count"] >= 1
    assert summary["excluded_case_count"] >= 1
