from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_structure_stability_reparameterization import (
    build_structure_stability_reparameterization_summary,
)


def test_structure_stability_reparameterization_bounds() -> None:
    summary = build_structure_stability_reparameterization_summary()
    hm = summary["headline_metrics"]

    assert hm["stability_intensity"] > 0.0
    assert 0.0 < hm["stability_strength"] < 1.0
    assert hm["closure_alignment"] > 0.0
    assert hm["stability_balance"] > hm["stability_strength"]
