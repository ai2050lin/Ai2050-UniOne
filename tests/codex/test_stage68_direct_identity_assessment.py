from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage68_direct_identity_assessment import build_direct_identity_assessment_summary


def test_stage68_direct_identity_assessment() -> None:
    summary = build_direct_identity_assessment_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["direct_closure"] > 0.76
    assert hm["direct_falsifiability"] > 0.76
    assert hm["direct_dependency_penalty"] < 0.29
    assert hm["direct_identity_readiness"] > 0.77
    assert status["status_short"] == "near_first_principles_theory"

    out_path = ROOT / "tests" / "codex_temp" / "stage68_direct_identity_assessment_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
