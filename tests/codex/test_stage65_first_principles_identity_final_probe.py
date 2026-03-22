from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage65_first_principles_identity_final_probe import build_first_principles_identity_final_probe_summary


def test_stage65_first_principles_identity_final_probe() -> None:
    summary = build_first_principles_identity_final_probe_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["final_closure"] > 0.69
    assert hm["final_falsifiability"] > 0.70
    assert hm["final_dependency_penalty"] < 0.42
    assert hm["final_identity_readiness"] > 0.67
    assert status["status_short"] == "phenomenological_transition"

    out_path = ROOT / "tests" / "codex_temp" / "stage65_first_principles_identity_final_probe_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
