from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage66_primitive_metric_decomposition import build_primitive_metric_decomposition_summary


def test_stage66_primitive_metric_decomposition() -> None:
    summary = build_primitive_metric_decomposition_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["primitive_decomposition_score"] > 0.779
    assert hm["native_metric_closure"] > 0.71
    assert hm["primitive_reconstruction_error"] < 0.25
    assert status["status_short"] == "primitive_decomposition_active"

    out_path = ROOT / "tests" / "codex_temp" / "stage66_primitive_metric_decomposition_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
