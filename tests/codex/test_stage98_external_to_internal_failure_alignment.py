from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage98_external_to_internal_failure_alignment import build_external_to_internal_failure_alignment_summary


def test_stage98_external_to_internal_failure_alignment() -> None:
    summary = build_external_to_internal_failure_alignment_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["path_alignment_stability"] >= 0.90
    assert hm["receiver_alignment_stability"] >= 0.75
    assert hm["weakest_clause_name"] in {
        "neuron_anchor_clause",
        "bundle_sync_clause",
        "field_compatibility_clause",
        "repair_transfer_clause",
        "evidence_isolation_clause",
    }
    assert hm["weakest_clause_score"] <= 0.55
    assert hm["clause_alignment_rate"] >= 0.50
    assert hm["alignment_coherence_mean"] >= 0.70
    assert hm["internal_external_gap"] <= 0.15
    assert hm["external_to_internal_alignment_score"] >= 0.75
    assert len(summary["alignment_records"]) == 8
    assert status["status_short"] in {
        "external_to_internal_failure_alignment_ready",
        "external_to_internal_failure_alignment_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage98_external_to_internal_failure_alignment_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
