from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage65_selector_master_equation_closure import build_selector_master_equation_closure_summary


def test_stage65_selector_master_equation_closure() -> None:
    summary = build_selector_master_equation_closure_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["master_equation_coherence"] > 0.79
    assert hm["master_equation_closure"] > 0.78
    assert hm["residual_master_gap"] < 0.22
    assert hm["equation_constraint_lock"] > 0.74
    assert status["status_short"] == "master_equation_nearly_closed"

    out_path = ROOT / "tests" / "codex_temp" / "stage65_selector_master_equation_closure_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
