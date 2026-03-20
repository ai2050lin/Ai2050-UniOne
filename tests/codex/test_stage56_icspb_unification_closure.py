from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_icspb_unification_closure import build_icspb_unification_closure_summary


def test_icspb_unification_closure_positive() -> None:
    summary = build_icspb_unification_closure_summary()
    hm = summary["headline_metrics"]

    assert 0.0 < hm["object_unification_strength"] < 1.0
    assert 0.0 < hm["transport_unification_strength"] < 1.0
    assert 0.0 < hm["remap_closure_core"] < 1.0
    assert 0.0 < hm["support_gap_reduced"] < 1.0
    assert 0.0 < hm["closure_stability"] < 1.0
