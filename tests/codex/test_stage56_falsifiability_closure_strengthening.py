from __future__ import annotations

from tests.codex.stage56_falsifiability_closure_strengthening import (
    build_falsifiability_closure_strengthening_summary,
)


def test_falsifiability_closure_strengthening_improves_closure() -> None:
    hm = build_falsifiability_closure_strengthening_summary()["headline_metrics"]
    assert hm["falsifiability_closure_stable"] > 0.4672124872704404
    assert hm["residual_nonfalsifiable_stable"] < 0.5327875127295596
