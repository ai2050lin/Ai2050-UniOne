from __future__ import annotations

from tests.codex.stage56_transport_unification_cross_version_validation import (
    build_transport_unification_cross_version_validation_summary,
)


def test_transport_unification_cross_version_validation_is_positive() -> None:
    hm = build_transport_unification_cross_version_validation_summary()["headline_metrics"]
    assert hm["feature_growth_consistency"] > 0.0
    assert hm["structure_growth_consistency"] > 0.0
    assert hm["cross_version_stability"] > 0.0
    assert hm["rollback_risk"] < 1.0
