from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage87_evidence_independence_audit import build_evidence_independence_audit_summary


def test_stage87_evidence_independence_audit() -> None:
    summary = build_evidence_independence_audit_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["independent_evidence_chain_exists"] is False
    assert hm["summary_backfeed_risk"] > 0.50
    assert hm["hardcoded_scenario_hits"] >= 3
    assert hm["handcrafted_law_hits"] >= 1
    assert hm["high_risk_count"] >= 3
    assert len(summary["audit_checks"]) == 8
    assert status["status_short"] in {
        "evidence_independence_audit_high_risk",
        "evidence_independence_audit_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage87_evidence_independence_audit_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
