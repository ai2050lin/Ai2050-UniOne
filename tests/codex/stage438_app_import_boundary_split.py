from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "frontend" / "src" / "App.jsx"
WORKSPACE_BRIDGE_PATH = ROOT / "frontend" / "src" / "blueprint" / "appleNeuronWorkspaceBridge.js"
SCENE_BRIDGE_PATH = ROOT / "frontend" / "src" / "blueprint" / "appleNeuronSceneBridge.jsx"
INFO_BRIDGE_PATH = ROOT / "frontend" / "src" / "blueprint" / "appleNeuronInfoPanelsBridge.jsx"
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / f"stage438_app_import_boundary_split_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    app_source = APP_PATH.read_text(encoding="utf-8")

    checks = {
        "workspace_bridge_exists": WORKSPACE_BRIDGE_PATH.exists(),
        "scene_bridge_exists": SCENE_BRIDGE_PATH.exists(),
        "info_bridge_exists": INFO_BRIDGE_PATH.exists(),
        "app_uses_workspace_bridge": "from './blueprint/appleNeuronWorkspaceBridge'" in app_source,
        "app_uses_scene_bridge": "from './blueprint/appleNeuronSceneBridge'" in app_source,
        "app_uses_info_bridge": "from './blueprint/appleNeuronInfoPanelsBridge'" in app_source,
        "app_no_longer_directly_imports_old_bundle": "from './blueprint/AppleNeuron3DTab'" not in app_source,
    }

    summary = {
        "stage": "stage438_app_import_boundary_split",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "app": str(APP_PATH.relative_to(ROOT)),
            "workspace_bridge": str(WORKSPACE_BRIDGE_PATH.relative_to(ROOT)),
            "scene_bridge": str(SCENE_BRIDGE_PATH.relative_to(ROOT)),
            "info_bridge": str(INFO_BRIDGE_PATH.relative_to(ROOT)),
        },
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
