try:
    from tests.codex.stage56_encoding_circuit_formation import build_encoding_circuit_formation_summary
except ModuleNotFoundError:
    from stage56_encoding_circuit_formation import build_encoding_circuit_formation_summary


def test_encoding_circuit_formation_has_positive_margin() -> None:
    summary = build_encoding_circuit_formation_summary()
    hm = summary["headline_metrics"]
    assert hm["local_stimulation"] > hm["steady_state_pressure"]
    assert hm["circuit_margin"] > 0.0

