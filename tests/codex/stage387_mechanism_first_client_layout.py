from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / "stage387_mechanism_first_client_layout_20260325"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "Stage387",
        "title": "运行机制优先客户端布局",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "core_change": "把 3D 客户端从抽象聚合体优先改成层带、有效神经元点、传播路径、原始分析优先。",
        "view_priority": [
            "层带骨架",
            "名词有效神经元点",
            "共享承载与偏置偏转叠加层",
            "逐层传播路径",
            "原始分析叠加层",
        ],
        "display_rules": {
            "aggregate_nodes": "弱化，只保留小环或细柱，不再用大球主导画面",
            "raw_points": "增强，原始点云成为主视觉层",
            "relay_paths": "增强，路径带比聚合体更显眼",
            "right_panel": "优先显示原始字段、参数位和来源阶段",
        },
        "expected_effect": [
            "用户先看到真实运行层级，再看到结构概括",
            "有效神经元和原始点不再被抽象大球压住",
            "共享承载、偏置偏转、逐层放大三段链更直观",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
