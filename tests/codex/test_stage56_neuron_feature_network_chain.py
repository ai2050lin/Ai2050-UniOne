from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_neuron_feature_network_chain import build_neuron_feature_network_chain_summary


def test_neuron_feature_network_chain_positive() -> None:
    summary = build_neuron_feature_network_chain_summary()
    hm = summary["headline_metrics"]

    assert hm["neuron_seed_signal"] > 0.0
    assert hm["feature_selection_signal"] > 0.0
    assert hm["feature_lock_signal"] > hm["feature_selection_signal"]
    assert hm["network_growth_signal"] > hm["circuit_closure_signal"]
    assert hm["chain_margin"] > hm["network_growth_signal"]
