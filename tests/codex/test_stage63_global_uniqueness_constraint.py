from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage63_global_uniqueness_constraint import build_global_uniqueness_constraint_summary


def test_stage63_global_uniqueness_constraint() -> None:
    summary = build_global_uniqueness_constraint_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["global_uniqueness_score"] > 0.84
    assert hm["mathematical_uniqueness_score"] > 0.82
    assert hm["unique_selector_constraint"] > 0.81
    assert status["status_short"] == "global_uniqueness_strongly_supported"

    out_path = ROOT / "tests" / "codex_temp" / "stage63_global_uniqueness_constraint_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
