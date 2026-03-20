from __future__ import annotations

from tests.codex.stage56_high_retention_cross_version_validation import build_high_retention_cross_version_validation_summary


def test_high_retention_cross_version_validation_is_stable() -> None:
    hm = build_high_retention_cross_version_validation_summary()["headline_metrics"]
    assert hm["cross_keep_core"] > 0.4450256466657768
    assert hm["cross_keep_floor"] > 0.4023231667156446
    assert hm["cross_keep_margin"] >= 0.0
