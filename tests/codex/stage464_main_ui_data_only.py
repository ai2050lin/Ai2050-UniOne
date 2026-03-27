from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
ROADMAP_PATH = ROOT / "frontend" / "src" / "blueprint" / "ProjectRoadmapTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage464_main_ui_data_only_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    panel_source = PANEL_PATH.read_text(encoding="utf-8")
    apple_source = APPLE_PATH.read_text(encoding="utf-8")
    roadmap_source = ROADMAP_PATH.read_text(encoding="utf-8")

    checks = {
        "panel_no_longer_sets_theory_object": "workspace.setTheoryObject(" not in panel_source,
        "panel_no_longer_sets_analysis_mode": "workspace.setAnalysisMode(" not in panel_source,
        "panel_mentions_strategy_migration": "统一移动到战略层级路线图" in panel_source,
        "apple_main_ui_removes_theory_object_selector": "onClick={() => setTheoryObject(item.id)}" not in apple_source,
        "apple_main_ui_has_moved_theory_notice": "理论分析已迁出主界面" in apple_source,
        "apple_data_metrics_no_longer_show_icspb_object": "ICSPB 对象:" not in apple_source,
        "apple_artifact_preview_is_data_named": "数据摘要与查看提示" in apple_source,
        "roadmap_has_theory_host_section": "理论分析承载区" in roadmap_source,
        "roadmap_mentions_main_ui_data_only": "主界面现在只保留数据观察、差异比较、样本验证和回放" in roadmap_source,
    }

    summary = {
        "stage": "stage464_main_ui_data_only",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "panel": str(PANEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "apple_tab": str(APPLE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "roadmap": str(ROADMAP_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
