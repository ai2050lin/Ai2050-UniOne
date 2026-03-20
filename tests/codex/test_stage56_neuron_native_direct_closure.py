from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_neuron_native_direct_closure import build_neuron_native_direct_closure_summary


def test_neuron_native_direct_closure_positive() -> None:
    summary = build_neuron_native_direct_closure_summary()
    hm = summary["headline_metrics"]

    assert hm["neuron_seed_direct"] > 0.0
    assert hm["neuron_select_direct"] > 0.0
    assert hm["neuron_lock_direct"] > hm["neuron_select_direct"]
    assert hm["neuron_native_core"] > hm["neuron_lock_direct"]
    assert 0.0 < hm["neuron_closure_confidence"] < 1.0
