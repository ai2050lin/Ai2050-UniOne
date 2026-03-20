try:
    from tests.codex.stage56_neurodynamics_bridge_v4 import build_neurodynamics_bridge_v4_summary
except ModuleNotFoundError:
    from stage56_neurodynamics_bridge_v4 import build_neurodynamics_bridge_v4_summary


def test_neurodynamics_bridge_v4_has_positive_margin() -> None:
    summary = build_neurodynamics_bridge_v4_summary()
    hm = summary["headline_metrics"]
    assert hm["local_excitation"] > hm["competitive_inhibition"]
    assert hm["dynamic_margin"] > 0.0
