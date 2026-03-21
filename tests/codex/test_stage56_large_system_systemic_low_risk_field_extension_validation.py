from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_large_system_systemic_low_risk_field_extension_validation import (
    build_large_system_systemic_low_risk_field_extension_validation_summary,
)


def test_stage56_large_system_systemic_low_risk_field_extension_validation() -> None:
    summary = build_large_system_systemic_low_risk_field_extension_validation_summary()
    hm = summary["headline_metrics"]

    for key in [
        "systemic_low_risk_field_strength",
        "systemic_low_risk_field_structure",
        "systemic_low_risk_field_route",
        "systemic_low_risk_field_learning",
        "systemic_low_risk_field_penalty",
        "systemic_low_risk_field_readiness",
        "systemic_low_risk_field_score",
    ]:
        assert 0.0 <= hm[key] <= 1.0

    assert hm["systemic_low_risk_field_margin"] > 0.0

    out_path = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_extension_validation_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["systemic_low_risk_field_score"] == hm["systemic_low_risk_field_score"]
