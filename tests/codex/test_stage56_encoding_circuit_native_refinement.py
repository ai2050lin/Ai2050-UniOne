try:
    from tests.codex.stage56_encoding_circuit_native_refinement import build_encoding_circuit_native_refinement_summary
except ModuleNotFoundError:
    from stage56_encoding_circuit_native_refinement import build_encoding_circuit_native_refinement_summary


def test_encoding_circuit_native_refinement_rebalances_components() -> None:
    summary = build_encoding_circuit_native_refinement_summary()
    hm = summary["headline_metrics"]
    assert hm["seed_refined"] > hm["bind_refined"] > hm["embed_refined"] > 0.0
    assert hm["pressure_refined"] > 0.0
    assert hm["encode_balance_refined"] > 0.0

