from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / f"stage434_language_panel_foundation_rewrite_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "has_basic_info_section": "基础信息" in source,
        "has_animation_control_section": "动画控制" in source,
        "has_five_layer_section": "五层测试体系" in source,
        "has_basic_encoding_entry": "基础编码" in source,
        "has_clean_research_layer_labels": all(
            text in source
            for text in [
                "静态编码层",
                "动态路径层",
                "结果回收层",
                "传播编码层",
                "语义角色层",
            ]
        ),
        "legacy_advanced_tools_removed": "高级分析工具" not in source,
    }

    ordered_titles = [
        "<span>基础信息</span>",
        "<span>动画控制</span>",
        "<span>五层测试体系</span>"
    ]
    title_positions = {title: source.find(title) for title in ordered_titles}
    checks["foundation_order_is_correct"] = all(
        title_positions[ordered_titles[i]] < title_positions[ordered_titles[i + 1]]
        for i in range(len(ordered_titles) - 1)
    )

    summary = {
        "stage": "stage434_language_panel_foundation_rewrite",
        "all_passed": all(checks.values()),
        "checks": checks,
        "title_positions": title_positions,
        "panel_path": str(PANEL_PATH.relative_to(ROOT)),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
