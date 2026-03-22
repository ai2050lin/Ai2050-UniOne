from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_native_observability_bridge import build_native_observability_bridge_summary


def test_stage70_native_observability_bridge() -> None:
    summary = build_native_observability_bridge_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["base_observability"] > 0.70
    assert hm["observability_bridge_score"] > 0.73
    assert hm["proxy_traceability_score"] > 0.79
    assert hm["hidden_proxy_gap"] < 0.23
    assert status["status_short"] == "native_observability_bridge_active"

    out_path = ROOT / "tests" / "codex_temp" / "stage70_native_observability_bridge_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
