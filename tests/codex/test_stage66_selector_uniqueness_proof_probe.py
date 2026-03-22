from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage66_selector_uniqueness_proof_probe import build_selector_uniqueness_proof_probe_summary


def test_stage66_selector_uniqueness_proof_probe() -> None:
    summary = build_selector_uniqueness_proof_probe_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["existence_support"] > 0.77
    assert hm["uniqueness_support"] > 0.78
    assert hm["stability_support"] > 0.76
    assert hm["proof_readiness"] > 0.78
    assert hm["proof_gap"] < 0.22
    assert status["status_short"] == "uniqueness_proof_probe_strengthened"

    out_path = ROOT / "tests" / "codex_temp" / "stage66_selector_uniqueness_proof_probe_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
