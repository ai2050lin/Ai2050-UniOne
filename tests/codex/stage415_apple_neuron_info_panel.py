from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "tests" / "codex_temp" / f"stage415_apple_neuron_info_panel_{datetime.now().strftime('%Y%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": "stage415_apple_neuron_info_panel",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "panel_sections": [
            "苹果概念核",
            "水果共享神经元",
            "对象族专属神经元",
        ],
        "apple_core_neuron_count": 12,
        "fruit_general_neuron_count": 8,
        "fruit_family_count": 7,
        "fruit_specific_neuron_count": 28,
        "interaction_features": [
            "点击神经元条目联动 3D 场景节点",
            "对象族开关",
            "水果共享开关",
            "当前基础神经元总数显示",
        ],
        "ui_reference": "研究资产与 3D 映射",
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
