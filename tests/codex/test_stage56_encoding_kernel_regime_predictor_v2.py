try:
    from tests.codex.stage56_encoding_kernel_regime_predictor_v2 import build_encoding_kernel_regime_predictor_v2_summary
except ModuleNotFoundError:
    from stage56_encoding_kernel_regime_predictor_v2 import build_encoding_kernel_regime_predictor_v2_summary


def test_encoding_kernel_regime_predictor_v2_improves_accuracy() -> None:
    summary = build_encoding_kernel_regime_predictor_v2_summary()
    assert summary["case_count"] >= 6
    assert summary["match_ratio"] >= 0.8

