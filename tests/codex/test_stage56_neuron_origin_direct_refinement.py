from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_neuron_origin_direct_refinement import build_neuron_origin_direct_refinement_summary


def test_neuron_origin_direct_refinement_positive() -> None:
    summary = build_neuron_origin_direct_refinement_summary()
    hm = summary["headline_metrics"]

    assert hm["origin_source_refined"] > 0.0
    assert hm["origin_focus_refined"] > 0.0
    assert hm["origin_retention_refined"] > 0.0
    assert hm["neuron_origin_margin_v2"] > hm["origin_source_refined"]
    assert 0.0 < hm["origin_stability_v2"] < 1.0
