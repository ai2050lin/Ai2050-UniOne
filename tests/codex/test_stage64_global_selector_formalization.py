from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage64_global_selector_formalization import build_global_selector_formalization_summary


def test_stage64_global_selector_formalization() -> None:
    summary = build_global_selector_formalization_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["selector_energy_coherence"] > 0.82
    assert hm["selector_formalization_score"] > 0.77
    assert hm["selector_closure"] > 0.80
    assert hm["residual_selector_gap"] < 0.21
    assert status["status_short"] == "selector_formalized_not_closed"

    out_path = ROOT / "tests" / "codex_temp" / "stage64_global_selector_formalization_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
