from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage96_independent_evidence_core import build_independent_evidence_core_summary


def test_stage96_independent_evidence_core() -> None:
    summary = build_independent_evidence_core_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["anchor_independence_strength"] >= 0.85
    assert hm["external_refutation_support"] >= 0.75
    assert hm["backfeed_suppression_strength"] >= 0.15
    assert hm["cross_plane_consistency"] >= 0.60
    assert hm["independent_ready_gap"] <= 0.85
    assert hm["independent_evidence_core_score"] >= 0.55
    assert len(summary["plane_records"]) == 4
    assert status["status_short"] in {
        "independent_evidence_core_ready",
        "independent_evidence_core_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage96_independent_evidence_core_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
