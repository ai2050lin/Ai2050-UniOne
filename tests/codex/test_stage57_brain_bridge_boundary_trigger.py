from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_brain_bridge_boundary_trigger import build_brain_bridge_boundary_trigger_summary


def test_stage57_brain_bridge_boundary_trigger() -> None:
    summary = build_brain_bridge_boundary_trigger_summary()
    hm = summary["headline_metrics"]

    assert hm["stressed_direct_structure"] < 0.78
    assert hm["stressed_direct_route"] < 0.79
    assert hm["stressed_shared_red_reuse"] < 0.80
    assert hm["stressed_brain_gap"] > 0.18
    assert hm["bridge_boundary_readiness"] < 0.80
    assert summary["bridge_trigger"]["triggered"] is True

    out_path = ROOT / "tests" / "codex_temp" / "stage57_brain_bridge_boundary_trigger_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["bridge_trigger"]["triggered"] is True
