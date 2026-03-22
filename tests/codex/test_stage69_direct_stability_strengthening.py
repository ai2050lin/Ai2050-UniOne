from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage69_direct_stability_strengthening import build_direct_stability_strengthening_summary


def test_stage69_direct_stability_strengthening() -> None:
    summary = build_direct_stability_strengthening_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["stability_gain"] > 0.78
    assert hm["strengthened_direct_stability_support"] > 0.75
    assert hm["residual_stability_gap"] < 0.25
    assert status["status_short"] == "direct_stability_strengthened"

    out_path = ROOT / "tests" / "codex_temp" / "stage69_direct_stability_strengthening_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
