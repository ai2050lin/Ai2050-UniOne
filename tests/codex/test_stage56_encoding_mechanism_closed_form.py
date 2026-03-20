try:
    from tests.codex.stage56_encoding_mechanism_closed_form import build_encoding_mechanism_closed_form_summary
except ModuleNotFoundError:
    from stage56_encoding_mechanism_closed_form import build_encoding_mechanism_closed_form_summary


def test_encoding_mechanism_closed_form_positive_margin() -> None:
    summary = build_encoding_mechanism_closed_form_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_core"] > 0.0
    assert hm["closed_form_margin"] > 0.0

