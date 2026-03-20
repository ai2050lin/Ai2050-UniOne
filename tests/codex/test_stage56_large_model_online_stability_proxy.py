try:
    from tests.codex.stage56_large_model_online_stability_proxy import build_large_model_stability_summary
except ModuleNotFoundError:
    from stage56_large_model_online_stability_proxy import build_large_model_stability_summary


def test_large_model_stability_summary_has_cases() -> None:
    summary = build_large_model_stability_summary()
    assert summary["case_count"] >= 6
    assert "headline_metrics" in summary
    assert summary["headline_metrics"]["best_plasticity_case"]
