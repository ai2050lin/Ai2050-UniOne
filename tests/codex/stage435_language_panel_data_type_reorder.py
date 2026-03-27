from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / f"stage435_language_panel_data_type_reorder_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "has_data_type_section": "<span>数据类型</span>" in source,
        "has_basic_encoding_section": "<span>基础编码</span>" in source,
        "has_animation_control_section": "<span>动画控制</span>" in source,
        "basic_problem_removed": "基础伤口" not in source,
        "five_layer_label_removed": "五层测试体系" not in source,
    }

    positions = {
        "data_type": source.find("<span>数据类型</span>"),
        "basic_encoding": source.find("<span>基础编码</span>"),
        "animation_control": source.find("<span>动画控制</span>"),
    }
    checks["order_is_correct"] = (
        positions["data_type"] != -1
        and positions["basic_encoding"] != -1
        and positions["animation_control"] != -1
        and positions["data_type"] < positions["basic_encoding"] < positions["animation_control"]
    )

    summary = {
        "stage": "stage435_language_panel_data_type_reorder",
        "all_passed": all(checks.values()),
        "checks": checks,
        "positions": positions,
        "panel_path": str(PANEL_PATH.relative_to(ROOT)),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
