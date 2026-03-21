from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_local_generative_law_emergence import build_local_generative_law_emergence_summary


def test_stage56_local_generative_law_emergence() -> None:
    summary = build_local_generative_law_emergence_summary()
    hm = summary["headline_metrics"]

    assert 0.0 <= hm["patch_coherence"] <= 1.0
    assert 0.0 <= hm["fiber_reuse"] <= 1.0
    assert 0.0 <= hm["route_separation"] <= 1.0
    assert 0.0 <= hm["pressure_balance"] <= 1.0
    assert hm["local_law_emergence_score"] > 0.5
    assert hm["derivability_score"] > 0.5

    out_path = ROOT / "tests" / "codex_temp" / "stage56_local_generative_law_emergence_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["derivability_score"] == hm["derivability_score"]
