from __future__ import annotations

import json
from pathlib import Path

from stage100_backfeed_suppression_hardening import build_backfeed_suppression_hardening_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage100_backfeed_suppression_hardening() -> None:
    summary = build_backfeed_suppression_hardening_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["summary_backfeed_risk_before"] >= 0.90
    assert hm["summary_backfeed_risk_after"] <= hm["summary_backfeed_risk_before"]
    assert hm["suppression_gain"] >= 0.15
    assert hm["direct_raw_source_strength"] >= 0.70
    assert hm["raw_trace_alignment"] >= 0.75
    assert hm["evidence_isolation_support"] >= 0.25
    assert hm["legacy_dependency_penalty"] >= 0.70
    assert hm["hardened_backfeed_suppression_strength"] >= 0.45
    assert hm["backfeed_suppression_hardening_score"] >= 0.58
    assert len(summary["stage_records"]) == 3
    assert status["status_short"] in {
        "backfeed_suppression_hardening_ready",
        "backfeed_suppression_hardening_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage100_backfeed_suppression_hardening_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
