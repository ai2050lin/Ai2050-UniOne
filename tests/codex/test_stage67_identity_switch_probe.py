from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage67_identity_switch_probe import build_identity_switch_probe_summary


def test_stage67_identity_switch_probe() -> None:
    summary = build_identity_switch_probe_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["switched_closure"] > 0.75
    assert hm["switched_falsifiability"] > 0.79
    assert hm["switched_dependency_penalty"] < 0.29
    assert hm["switched_identity_readiness"] > 0.77
    assert status["status_short"] == "near_first_principles_theory"

    out_path = ROOT / "tests" / "codex_temp" / "stage67_identity_switch_probe_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
