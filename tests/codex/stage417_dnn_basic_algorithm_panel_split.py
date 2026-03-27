from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "tests" / "codex_temp" / f"stage417_dnn_basic_algorithm_panel_split_{datetime.now().strftime('%Y%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": "stage417_dnn_basic_algorithm_panel_split",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "panel_tabs": [
            "基础信息",
            "算法显示",
        ],
        "default_display_levels": {
            "basic_neurons": True,
            "object_family": True,
            "parameter_state": False,
            "mechanism_chain": False,
            "advanced_analysis": False,
        },
        "manual_algorithm_actions": [
            "显示苹果概念核",
            "显示静态编码层",
            "显示运行链路",
        ],
        "principle": "先显示基础信息，再由用户手动打开苹果概念核、静态编码层等算法内容。",
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
