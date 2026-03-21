from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage58_local_law_necessity_scan import build_local_law_necessity_scan_summary


def test_stage58_local_law_necessity_scan() -> None:
    summary = build_local_law_necessity_scan_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["full_system_survives"] is True
    assert hm["ablated_survival_count"] == 0
    assert hm["necessity_count"] == 4
    assert hm["necessity_strength"] > 0.70
    assert hm["proof_gap"] > 0.15
    assert status["status_short"] == "necessity_supported_not_proven"
    assert "context_gate" in status["necessary_components"]

    out_path = ROOT / "tests" / "codex_temp" / "stage58_local_law_necessity_scan_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
