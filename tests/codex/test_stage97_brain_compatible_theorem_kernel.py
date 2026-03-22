from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage97_brain_compatible_theorem_kernel import build_brain_compatible_theorem_kernel_summary


def test_stage97_brain_compatible_theorem_kernel() -> None:
    summary = build_brain_compatible_theorem_kernel_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["neuron_anchor_clause"] >= 0.60
    assert hm["bundle_sync_clause"] >= 0.55
    assert hm["field_compatibility_clause"] >= 0.55
    assert hm["repair_transfer_clause"] >= 0.60
    assert hm["evidence_isolation_clause"] >= 0.25
    assert hm["theorem_viability_gap"] <= 0.75
    assert hm["brain_compatible_theorem_kernel_score"] >= 0.55
    assert len(summary["clause_records"]) == 5
    assert status["status_short"] in {
        "brain_compatible_theorem_kernel_ready",
        "brain_compatible_theorem_kernel_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage97_brain_compatible_theorem_kernel_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
