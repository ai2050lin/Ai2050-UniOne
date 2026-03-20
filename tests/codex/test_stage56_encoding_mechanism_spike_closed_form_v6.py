from __future__ import annotations

from stage56_encoding_mechanism_spike_closed_form_v6 import build_encoding_mechanism_spike_closed_form_v6_summary


def test_encoding_mechanism_spike_closed_form_v6_positive() -> None:
    summary = build_encoding_mechanism_spike_closed_form_v6_summary()
    hm = summary["headline_metrics"]
    assert hm["encoding_margin_v6"] > 0.0
    assert hm["structure_core_v6"] > hm["pressure_core_v6"]
