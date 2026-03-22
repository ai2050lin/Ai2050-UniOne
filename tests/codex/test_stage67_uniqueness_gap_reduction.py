from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage67_uniqueness_gap_reduction import build_uniqueness_gap_reduction_summary


def test_stage67_uniqueness_gap_reduction() -> None:
    summary = build_uniqueness_gap_reduction_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["reduced_existence_support"] > 0.87
    assert hm["reduced_uniqueness_support"] > 0.88
    assert hm["reduced_stability_support"] > 0.85
    assert hm["reduced_proof_readiness"] > 0.86
    assert hm["reduced_proof_gap"] < 0.14
    assert status["status_short"] == "uniqueness_gap_reduced"

    out_path = ROOT / "tests" / "codex_temp" / "stage67_uniqueness_gap_reduction_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
