from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage93_law_to_theorem_bridge import build_law_to_theorem_bridge_summary


def test_stage93_law_to_theorem_bridge() -> None:
    summary = build_law_to_theorem_bridge_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["premise_clause_strength"] >= 0.90
    assert hm["boundary_clause_strength"] >= 0.78
    assert hm["failure_clause_explicitness"] >= 0.95
    assert hm["brain_compatibility_clause"] >= 0.45
    assert hm["theorem_ready_gap"] <= 0.56
    assert hm["law_to_theorem_bridge_score"] >= 0.74
    assert len(summary["clause_records"]) == 4
    assert status["status_short"] in {
        "law_to_theorem_bridge_ready",
        "law_to_theorem_bridge_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage93_law_to_theorem_bridge_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
