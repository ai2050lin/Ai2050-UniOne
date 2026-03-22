from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage68_direct_signal_bundle import build_direct_signal_bundle_summary


def test_stage68_direct_signal_bundle() -> None:
    summary = build_direct_signal_bundle_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["direct_structural_coherence"] > 0.72
    assert hm["direct_task_recovery_support"] > 0.77
    assert hm["direct_boundary_resilience"] > 0.56
    assert hm["direct_weight_grounding"] > 0.84
    assert status["status_short"] == "direct_signal_bundle_ready"

    out_path = ROOT / "tests" / "codex_temp" / "stage68_direct_signal_bundle_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
