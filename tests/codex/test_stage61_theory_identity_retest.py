from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage61_theory_identity_retest import build_theory_identity_retest_summary


def test_stage61_theory_identity_retest() -> None:
    summary = build_theory_identity_retest_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["retest_closure"] > 0.58
    assert hm["retest_falsifiability"] > 0.72
    assert hm["retest_dependency_penalty"] < 0.62
    assert status["status_short"] == "phenomenological_transition"

    out_path = ROOT / "tests" / "codex_temp" / "stage61_theory_identity_retest_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
