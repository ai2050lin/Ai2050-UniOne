from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "tests" / "codex_temp" / f"stage418_dnn_panel_restructure_for_basic_concepts_{datetime.now().strftime('%Y%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": "stage418_dnn_panel_restructure_for_basic_concepts",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "panel_order": [
            "基础信息",
            "算法显示",
        ],
        "basic_info_capability": {
            "concept_browser": True,
            "concepts": ["apple", "fruit", "pear", "banana", "orange", "grape", "peach", "mango"],
            "node_click_link_to_scene": True,
        },
        "algorithm_capability": {
            "show_apple_core": True,
            "show_static_encoding": True,
            "show_runtime_chain": True,
        },
        "default_display_levels": {
            "basic_neurons": True,
            "object_family": True,
            "parameter_state": False,
            "mechanism_chain": False,
            "advanced_analysis": False,
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
