try:
    from tests.codex.stage56_encoding_mechanism_closed_form_v2 import build_encoding_mechanism_closed_form_v2_summary
except ModuleNotFoundError:
    from stage56_encoding_mechanism_closed_form_v2 import build_encoding_mechanism_closed_form_v2_summary


def test_encoding_mechanism_closed_form_v2_positive_margin() -> None:
    summary = build_encoding_mechanism_closed_form_v2_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_kernel_v2"] > 0.0
    assert hm["closed_form_margin_v2"] > hm["encoding_kernel_v2"]

