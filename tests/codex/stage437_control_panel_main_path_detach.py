from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "frontend" / "src" / "App.jsx"
APPLE_TAB_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / f"stage437_control_panel_main_path_detach_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    app_source = APP_PATH.read_text(encoding="utf-8")
    apple_tab_source = APPLE_TAB_PATH.read_text(encoding="utf-8")

    checks = {
        "app_no_control_panel_import": "AppleNeuronControlPanels" not in app_source,
        "apple_tab_imports_language_panel": "import LanguageResearchControlPanel from '../components/LanguageResearchControlPanel';" in apple_tab_source,
        "apple_tab_uses_language_panel": "LanguageResearchControlPanel workspace={workspace} structureTab=\"circuit\"" in apple_tab_source,
        "apple_tab_main_path_no_control_panel_render": "<AppleNeuronControlPanels workspace={workspace} />" not in apple_tab_source,
    }

    summary = {
        "stage": "stage437_control_panel_main_path_detach",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "app": str(APP_PATH.relative_to(ROOT)),
            "apple_tab": str(APPLE_TAB_PATH.relative_to(ROOT)),
        },
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
