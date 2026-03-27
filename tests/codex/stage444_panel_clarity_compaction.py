from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage444_panel_clarity_compaction_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "has_focus_summary_title": "语言主线控制台" in source,
        "has_focus_summary_cards": "FocusSummaryItem" in source and "当前拼图" in source,
        "has_research_entry_section": "研究入口" in source,
        "has_compact_basic_encoding_grid": "DetailItem label=\"当前主视图\"" in source and "DetailItem label=\"动画状态\"" in source,
        "has_compact_puzzle_summary": "这里聚焦最弱轴、当前拼图和下一步动作" in source,
        "has_puzzle_detail_grid": "DetailItem label=\"映射变量\"" in source and "DetailItem label=\"下一步动作\"" in source,
        "has_short_axis_all_label": ">\\n            全部\\n" in source or "全部" in source,
        "has_scrollable_puzzle_list": "overflowY: 'auto'" in source and "maxHeight: '220px'" in source,
    }

    summary = {
        "stage": "stage444_panel_clarity_compaction",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "panel": str(PANEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
