from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_neuron_origin_native_probe import build_neuron_origin_native_probe_summary


def test_neuron_origin_native_probe_positive() -> None:
    summary = build_neuron_origin_native_probe_summary()
    hm = summary["headline_metrics"]

    assert hm["pulse_source_strength"] > 0.0
    assert hm["selectivity_focus"] > 0.0
    assert hm["lock_retention"] > hm["selectivity_focus"]
    assert hm["neuron_origin_core"] > hm["pulse_source_strength"]
    assert 0.0 < hm["neuron_origin_confidence"] < 1.0
