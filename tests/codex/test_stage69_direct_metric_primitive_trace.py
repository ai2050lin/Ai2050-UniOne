from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage69_direct_metric_primitive_trace import build_direct_metric_primitive_trace_summary


def test_stage69_direct_metric_primitive_trace() -> None:
    summary = build_direct_metric_primitive_trace_summary()
    status = summary["status"]

    assert "P_patch" in summary["primitive_origin"]
    assert "direct_closure" in summary["identity_breakdown"]
    assert "direct_structural_coherence" in summary["signal_breakdown"]
    assert status["status_short"] == "direct_metric_trace_ready"

    out_path = ROOT / "tests" / "codex_temp" / "stage69_direct_metric_primitive_trace_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
