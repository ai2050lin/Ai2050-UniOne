from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage69_direct_theorem_gap_compression import build_direct_theorem_gap_compression_summary


def test_stage69_direct_theorem_gap_compression() -> None:
    summary = build_direct_theorem_gap_compression_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["compressed_direct_theorem_readiness"] > 0.79
    assert hm["compressed_direct_theorem_gap"] < 0.21
    assert status["status_short"] == "direct_theorem_gap_compressed"

    out_path = ROOT / "tests" / "codex_temp" / "stage69_direct_theorem_gap_compression_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
