try:
    from tests.codex.stage56_circuit_to_regime_predictor import build_circuit_to_regime_predictor_summary
except ModuleNotFoundError:
    from stage56_circuit_to_regime_predictor import build_circuit_to_regime_predictor_summary


def test_circuit_to_regime_predictor_beats_random() -> None:
    summary = build_circuit_to_regime_predictor_summary()
    assert summary["case_count"] >= 6
    assert summary["match_ratio"] >= 0.5

