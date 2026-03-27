from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "codex_temp" / f"stage397_layer_basic_runtime_controls_{datetime.now().strftime('%Y%m%d')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    summary = {
        "stage": "stage397",
        "focus": "在不改当前28个layer主视图的前提下，补基础效果而非抽象层效果",
        "frontend_files": [
            "frontend/src/blueprint/AppleNeuron3DTab.jsx",
            "frontend/src/blueprint/data/layer_parameter_state_overlay_v1.js",
        ],
        "hard_constraints": {
            "preserve_current_layer_structure": True,
            "preserve_current_layer_form": True,
            "advanced_overlays_default_disabled": True,
        },
        "basic_runtime_controls": {
            "button_count": 3,
            "buttons": ["开始动画", "结束动画", "重新播放"],
            "runtime_profiles": 5,
            "runtime_profile_keys": [
                "static_encoding",
                "dynamic_route",
                "result_recovery",
                "propagation_encoding",
                "semantic_roles",
            ],
        },
        "parameter_overlay_stats": {
            "total_parameter_nodes": 15,
            "total_chain_links": 10,
            "default_visible_strategy": "参数态节点优先，抽象叠加默认关闭",
        },
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
