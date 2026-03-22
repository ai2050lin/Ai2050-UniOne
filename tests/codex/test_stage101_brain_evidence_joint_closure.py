from __future__ import annotations

import json
from pathlib import Path

from stage101_brain_evidence_joint_closure import build_brain_evidence_joint_closure_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage101_brain_evidence_joint_closure() -> None:
    summary = build_brain_evidence_joint_closure_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["neuron_anchor_joint"] >= 0.45
    assert hm["bundle_sync_joint"] >= 0.45
    assert hm["field_observability_joint"] >= 0.45
    assert hm["evidence_isolation_joint"] >= 0.30
    assert hm["real_world_bridge_joint"] >= 0.65
    assert hm["weakest_joint_clause_name"] in {
        "neuron_anchor_joint",
        "bundle_sync_joint",
        "field_observability_joint",
        "evidence_isolation_joint",
        "real_world_bridge_joint",
    }
    assert hm["weakest_joint_clause_score"] >= 0.30
    assert hm["brain_evidence_joint_closure_gap"] <= 0.70
    assert hm["brain_evidence_joint_closure_score"] >= 0.52
    assert len(summary["clause_records"]) == 5
    assert status["status_short"] in {
        "brain_evidence_joint_closure_ready",
        "brain_evidence_joint_closure_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage101_brain_evidence_joint_closure_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
