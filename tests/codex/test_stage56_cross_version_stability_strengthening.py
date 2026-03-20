from __future__ import annotations

from tests.codex.stage56_cross_version_stability_strengthening import build_cross_version_stability_strengthening_summary


def test_cross_version_stability_strengthening_improves_baseline() -> None:
    hm = build_cross_version_stability_strengthening_summary()["headline_metrics"]
    assert hm["cross_version_stability_stable"] > 0.43622510734194736
    assert hm["rollback_risk_reduced"] < 0.5978406532745425
    assert hm["stability_gain"] > 0.0
