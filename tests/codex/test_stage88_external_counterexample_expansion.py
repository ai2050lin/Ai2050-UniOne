from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage88_external_counterexample_expansion import build_external_counterexample_expansion_summary


def test_stage88_external_counterexample_expansion() -> None:
    summary = build_external_counterexample_expansion_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["family_coverage"] >= 0.95
    assert hm["triggered_family_coverage"] >= 0.60
    assert hm["expanded_trigger_rate"] >= 0.60
    assert hm["strongest_refutation_strength"] >= 0.58
    assert hm["average_refutation_strength"] >= 0.48
    assert hm["external_counterexample_expansion_score"] >= 0.68
    assert len(summary["scenario_records"]) == 8
    assert status["status_short"] in {
        "external_counterexample_expansion_ready",
        "external_counterexample_expansion_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage88_external_counterexample_expansion_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
