from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage407_parameter_summary_overlay_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "review_id": "parameter_summary_overlay_v1",
        "summary_overlay_enabled": True,
        "summary_fields": [
            "layer",
            "dimIndex",
            "value",
        ],
        "summary_position": [10.8, 8.4, 0],
        "target_view": "当前 28 层主视图",
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
