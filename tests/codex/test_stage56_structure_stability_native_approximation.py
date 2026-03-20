from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_structure_stability_native_approximation import (
    build_structure_stability_native_approximation_summary,
)


def test_structure_stability_native_approximation_positive() -> None:
    summary = build_structure_stability_native_approximation_summary()
    hm = summary["headline_metrics"]

    assert hm["native_stability_seed"] > 0.0
    assert hm["native_stability_binding"] > 0.0
    assert hm["native_stability_feedback"] > 0.0
    assert hm["native_stability_core"] > hm["native_stability_binding"]
    assert hm["native_stability_ratio"] > 0.0
