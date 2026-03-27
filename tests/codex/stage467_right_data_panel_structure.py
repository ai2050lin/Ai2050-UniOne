from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "frontend" / "src" / "App.jsx"
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchDataPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage467_right_data_panel_structure_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    app_source = APP_PATH.read_text(encoding="utf-8")
    panel_source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
      "component_exists": PANEL_PATH.exists(),
      "component_has_four_tabs": all(label in panel_source for label in ["当前焦点", "层数据", "样本回放", "资产与证据"]),
      "component_has_focus_summary": "当前焦点摘要" in panel_source,
      "component_has_replay_section": "样本回放详情" in panel_source,
      "component_has_asset_summary": "资产摘要" in panel_source,
      "app_imports_new_panel": "import LanguageResearchDataPanel from './components/LanguageResearchDataPanel';" in app_source,
      "app_uses_new_panel": "<LanguageResearchDataPanel" in app_source,
      "app_defaults_main_panel_to_focus": "setInfoPanelTab('focus');" in app_source,
      "app_uses_data_panel_title": "数据面板 · DNN /" in app_source,
    }

    summary = {
        "stage": "stage467_right_data_panel_structure",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "app": str(APP_PATH.relative_to(ROOT)).replace("\\", "/"),
            "panel": str(PANEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }

    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
