from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage65_completion_gap_attack import build_completion_gap_attack_summary


def test_stage65_completion_gap_attack() -> None:
    summary = build_completion_gap_attack_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["gap_reduction_gain"] > 0.69
    assert hm["attacked_completion_gap"] < 0.255
    assert hm["attacked_completion_readiness"] > 0.66
    assert hm["residual_completion_blocker"] < 0.42
    assert status["status_short"] == "completion_gap_under_attack"

    out_path = ROOT / "tests" / "codex_temp" / "stage65_completion_gap_attack_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
