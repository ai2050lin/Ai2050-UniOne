from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage85_external_counterexample_generator import build_external_counterexample_generator_summary


def test_stage85_external_counterexample_generator() -> None:
    summary = build_external_counterexample_generator_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["external_counterexample_diversity"] > 0.85
    assert hm["external_trigger_rate"] >= 0.20
    assert hm["strongest_refutation_strength"] > 0.55
    assert hm["shared_state_external_break_score"] > 0.55
    assert len(summary["scenario_records"]) == 5
    assert all(not item["derived_from_support_minus_constant"] for item in summary["scenario_records"])
    assert any(item["trigger_demonstrated"] for item in summary["scenario_records"])
    assert status["status_short"] in {
        "external_counterexample_generator_ready",
        "external_counterexample_generator_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage85_external_counterexample_generator_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
