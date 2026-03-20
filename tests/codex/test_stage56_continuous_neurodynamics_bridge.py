try:
    from tests.codex.stage56_continuous_neurodynamics_bridge import build_continuous_neurodynamics_bridge_summary
except ModuleNotFoundError:
    from stage56_continuous_neurodynamics_bridge import build_continuous_neurodynamics_bridge_summary


def test_continuous_neurodynamics_bridge_positive_balance() -> None:
    summary = build_continuous_neurodynamics_bridge_summary()
    hm = summary["headline_metrics"]
    assert hm["dV_dt"] > 0.0
    assert hm["dB_dt"] > 0.0
    assert hm["dynamic_balance"] > 0.0
