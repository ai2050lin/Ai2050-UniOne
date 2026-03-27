from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage460_remove_runtime_overlay_window_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = APPLE_PATH.read_text(encoding="utf-8")

    checks = {
        "scene_keeps_layer_guides": "<LayerGuides activeLayer={activeLayer} />" in source,
        "scene_no_longer_renders_runtime_window": "title={`${runtimeProfile.label} 基础动画`}" not in source,
        "runtime_control_component_still_exists": "function LayerBasicRuntimeControls(" in source,
        "scene_still_renders_mechanism_links": "displayLevels?.mechanism_chain !== false" in source,
    }

    summary = {
        "stage": "stage460_remove_runtime_overlay_window",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "apple_tab": str(APPLE_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
