from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
ROADMAP_PATH = ROOT / "frontend" / "src" / "blueprint" / "ProjectRoadmapTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage465_control_panel_operations_only_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    panel_source = PANEL_PATH.read_text(encoding="utf-8")
    roadmap_source = ROADMAP_PATH.read_text(encoding="utf-8")

    removed_headings = [
        ">语言主线控制台<",
        "<span>研究总览</span>",
        "<span>关键性质</span>",
        "<span>验证入口</span>",
        "<span>概念关联</span>",
        ">拼图对比台<",
    ]
    kept_labels = [
        "主界面操作入口",
        "研究入口",
        "基础编码",
        "基础拼图仓",
        "样本回放",
        "动画控制",
    ]
    roadmap_labels = [
        "主界面迁移内容",
        "语言主线控制台",
        "研究总览",
        "关键性质",
        "验证入口",
        "概念关联",
        "拼图对比台",
    ]

    checks = {
        "panel_removed_display_sections": all(label not in panel_source for label in removed_headings),
        "panel_keeps_operation_sections": all(label in panel_source for label in kept_labels),
        "panel_mentions_strategy_migration": "统一移动到战略层级路线图" in panel_source,
        "roadmap_has_migration_host": all(label in roadmap_source for label in roadmap_labels),
        "roadmap_explains_control_panel_scope": "控制面板现在只保留操作入口" in roadmap_source,
    }

    summary = {
        "stage": "stage465_control_panel_operations_only",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "panel": str(PANEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "roadmap": str(ROADMAP_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }

    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
