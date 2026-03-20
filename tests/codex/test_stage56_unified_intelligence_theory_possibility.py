from __future__ import annotations

from tests.codex.stage56_unified_intelligence_theory_possibility import (
    build_unified_intelligence_theory_possibility_summary,
)


def test_unified_intelligence_theory_possibility_is_bounded() -> None:
    hm = build_unified_intelligence_theory_possibility_summary()["headline_metrics"]
    assert hm["unification_core"] > 0.0
    assert 0.0 <= hm["higher_unified_intelligence_possibility"] <= 1.0
    assert hm["first_principles_distance"] >= 0.0
    assert hm["falsifiability_gap"] >= 0.0
