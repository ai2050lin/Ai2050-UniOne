from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage64_completion_pathway_map import build_completion_pathway_map_summary


def test_stage64_completion_pathway_map() -> None:
    summary = build_completion_pathway_map_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["final_completion_readiness"] > 0.60
    assert hm["remaining_completion_gap"] < 0.41
    assert hm["pathway_confidence"] > 0.70
    assert hm["remaining_key_steps"] == 2
    assert status["status_short"] == "completion_path_visible_not_finished"

    out_path = ROOT / "tests" / "codex_temp" / "stage64_completion_pathway_map_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
