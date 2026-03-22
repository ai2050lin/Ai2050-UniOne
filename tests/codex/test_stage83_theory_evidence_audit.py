from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage83_theory_evidence_audit import build_theory_evidence_audit_summary


def test_stage83_theory_evidence_audit() -> None:
    summary = build_theory_evidence_audit_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["stage71_summary_dependency_fan_in"] >= 10
    assert hm["stage80_hardcoded_scenario_count"] == 5
    assert hm["stage82_hardcoded_law_count"] == 3
    assert hm["stage82_best_law_name"] == "sqrt"
    assert hm["stage82_best_law_margin"] < 0.01
    assert hm["roundtrip_only_test_count"] >= 4
    assert hm["derived_falsification_flag"] is True
    assert hm["best_law_fragility_flag"] is True
    assert hm["status_label_mismatch_flag"] is True
    assert hm["evidence_independence_score"] < 0.50
    assert hm["test_strength_score"] < 0.20
    assert hm["theory_correctness_confidence"] < 0.55
    assert status["status_short"] == "unproven_explanatory_framework"

    out_path = ROOT / "tests" / "codex_temp" / "stage83_theory_evidence_audit_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
