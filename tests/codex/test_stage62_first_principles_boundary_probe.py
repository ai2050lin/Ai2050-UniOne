from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_first_principles_boundary_probe import build_first_principles_boundary_probe_summary


def test_stage62_first_principles_boundary_probe() -> None:
    summary = build_first_principles_boundary_probe_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["first_principles_readiness"] > 0.65
    assert hm["distance_to_first_principles_theory"] > 0.08
    assert hm["remaining_boundary_count"] >= 4
    assert status["status_short"] == "phenomenological_transition"

    out_path = ROOT / "tests" / "codex_temp" / "stage62_first_principles_boundary_probe_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
