try:
    from tests.codex.stage56_stability_regime_map import build_regime_map_summary
except ModuleNotFoundError:
    from stage56_stability_regime_map import build_regime_map_summary


def test_regime_map_summary_has_rows() -> None:
    summary = build_regime_map_summary()
    assert summary["case_count"] >= 6
    assert sum(summary["headline_metrics"].values()) == summary["case_count"]

