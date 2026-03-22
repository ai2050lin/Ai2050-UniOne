from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage78_distributed_route_native_observability import build_distributed_route_native_observability_summary


def test_stage78_distributed_route_native_observability() -> None:
    summary = build_distributed_route_native_observability_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["distributed_route_traceability"] > 0.78
    assert hm["route_conflict_native_measure"] > 0.76
    assert hm["route_counterexample_triggerability"] > 0.79
    assert hm["field_proxy_gap"] < 0.21
    assert hm["route_native_observability_score"] > 0.79
    assert len(summary["scenario_records"]) == 3
    assert status["status_short"] in {
        "distributed_route_native_observable",
        "distributed_route_native_observability_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage78_distributed_route_native_observability_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
