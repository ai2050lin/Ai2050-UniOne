from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_large_system_systemic_low_risk_field_lattice_validation import (
    build_large_system_systemic_low_risk_field_lattice_validation_summary,
)


def test_stage56_large_system_systemic_low_risk_field_lattice_validation() -> None:
    summary = build_large_system_systemic_low_risk_field_lattice_validation_summary()
    hm = summary["headline_metrics"]

    assert 0.0 < hm["systemic_low_risk_field_lattice"] <= 1.0
    assert 0.0 < hm["systemic_low_risk_field_structure_lattice"] <= 1.0
    assert 0.0 < hm["systemic_low_risk_field_route_lattice"] <= 1.0
    assert 0.0 < hm["systemic_low_risk_field_learning_lattice"] <= 1.0
    assert 0.0 <= hm["systemic_low_risk_field_lattice_penalty"] < 1.0
    assert hm["systemic_low_risk_field_lattice_margin"] > 0.0

    out_path = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_lattice_validation_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["systemic_low_risk_field_lattice_score"] == hm["systemic_low_risk_field_lattice_score"]
