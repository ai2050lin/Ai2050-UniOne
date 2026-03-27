from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "frontend" / "src" / "App.jsx"
LANGUAGE_PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
BASIC_PANEL_PATH = ROOT / "frontend" / "src" / "components" / "BasicEncodingPanel.jsx"
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / f"stage436_basic_encoding_panel_detach_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    app_source = APP_PATH.read_text(encoding="utf-8")
    language_panel_source = LANGUAGE_PANEL_PATH.read_text(encoding="utf-8")
    basic_panel_source = BASIC_PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "basic_panel_exists": BASIC_PANEL_PATH.exists(),
        "app_no_longer_imports_control_panels": "AppleNeuronControlPanels" not in app_source,
        "language_panel_imports_basic_panel": "import BasicEncodingPanel from './BasicEncodingPanel';" in language_panel_source,
        "language_panel_embeds_basic_panel": "<BasicEncodingPanel workspace={workspace} />" in language_panel_source,
        "basic_panel_has_manual_input": "生成 3D 模型" in basic_panel_source,
        "basic_panel_has_asset_import": "导入并映射到 3D" in basic_panel_source,
    }

    summary = {
        "stage": "stage436_basic_encoding_panel_detach",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "app": str(APP_PATH.relative_to(ROOT)),
            "language_panel": str(LANGUAGE_PANEL_PATH.relative_to(ROOT)),
            "basic_panel": str(BASIC_PANEL_PATH.relative_to(ROOT)),
        },
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
