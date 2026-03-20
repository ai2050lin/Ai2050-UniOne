try:
    from tests.codex.stage56_encoding_circuit_native_variables import build_encoding_circuit_native_variable_summary
except ModuleNotFoundError:
    from stage56_encoding_circuit_native_variables import build_encoding_circuit_native_variable_summary


def test_encoding_circuit_native_variables_shape() -> None:
    summary = build_encoding_circuit_native_variable_summary()
    hm = summary["headline_metrics"]
    assert hm["seed_native"] > hm["embed_native"] > 0.0
    assert hm["encode_balance_native"] > 0.0

