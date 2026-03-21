from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage59_local_law_symbolic_derivation import build_local_law_symbolic_derivation_summary


def test_stage59_local_law_symbolic_derivation() -> None:
    summary = build_local_law_symbolic_derivation_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["symbolic_component_coverage"] == 1.0
    assert hm["symbolic_bridge_score"] > 0.70
    assert hm["symbolic_closure"] > 0.65
    assert hm["theorem_gap"] > 0.20
    assert status["status_short"] == "symbolic_bridge_not_closed"

    out_path = ROOT / "tests" / "codex_temp" / "stage59_local_law_symbolic_derivation_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
