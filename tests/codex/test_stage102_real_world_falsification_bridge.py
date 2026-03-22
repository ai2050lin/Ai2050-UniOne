from __future__ import annotations

import json
from pathlib import Path

from stage102_real_world_falsification_bridge import build_real_world_falsification_bridge_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage102_real_world_falsification_bridge() -> None:
    summary = build_real_world_falsification_bridge_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["task_context_bridge_strength"] >= 0.25
    assert hm["multiseed_probe_stability"] >= 0.60
    assert hm["bridge_alignment_support"] >= 0.65
    assert hm["falsification_triggerability"] >= 0.50
    assert hm["remaining_real_world_gap"] <= 0.75
    assert hm["real_world_falsification_bridge_score"] >= 0.58
    assert len(summary["dimension_records"]) == 3
    assert status["status_short"] in {
        "real_world_falsification_bridge_ready",
        "real_world_falsification_bridge_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage102_real_world_falsification_bridge_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
