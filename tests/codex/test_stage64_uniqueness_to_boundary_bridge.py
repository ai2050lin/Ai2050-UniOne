from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage64_uniqueness_to_boundary_bridge import build_uniqueness_to_boundary_bridge_summary


def test_stage64_uniqueness_to_boundary_bridge() -> None:
    summary = build_uniqueness_to_boundary_bridge_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["bridged_boundary_closure"] > 0.69
    assert hm["bridged_boundary_falsifiability"] > 0.76
    assert hm["bridged_dependency_penalty"] < 0.42
    assert hm["remaining_boundary_count"] == 0
    assert status["status_short"] == "uniqueness_boundary_bridge_active"

    out_path = ROOT / "tests" / "codex_temp" / "stage64_uniqueness_to_boundary_bridge_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
