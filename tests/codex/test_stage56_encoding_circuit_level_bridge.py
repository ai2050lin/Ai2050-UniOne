from __future__ import annotations

try:
    from tests.codex.stage56_encoding_circuit_level_bridge import (
        build_encoding_circuit_level_bridge_summary,
    )
except ModuleNotFoundError:
    from stage56_encoding_circuit_level_bridge import build_encoding_circuit_level_bridge_summary


def test_circuit_level_margin_positive() -> None:
    summary = build_encoding_circuit_level_bridge_summary()
    hm = summary["headline_metrics"]
    assert hm["excitatory_seed"] > 0.0
    assert hm["synchrony_binding"] >= 0.0
    assert hm["embedding_recruitment"] >= 0.0
    assert hm["inhibitory_pressure"] >= 0.0
    assert hm["circuit_level_margin"] > 0.0
