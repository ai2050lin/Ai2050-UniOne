from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_native_variable_improvement_audit import build_native_variable_improvement_audit_summary


def test_stage70_native_variable_improvement_audit() -> None:
    summary = build_native_variable_improvement_audit_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["direct_explainability_gain"] > 0.82
    assert hm["dependency_interpretability_gain"] > 0.59
    assert hm["metric_traceability_gain"] > 0.90
    assert hm["theorem_transparency_gain"] > 0.77
    assert hm["overall_native_improvement"] > 0.79
    assert status["status_short"] == "native_improvement_audited"

    out_path = ROOT / "tests" / "codex_temp" / "stage70_native_variable_improvement_audit_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
