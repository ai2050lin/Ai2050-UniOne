from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage458_layer_sweep_highlight_restore_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = APPLE_PATH.read_text(encoding="utf-8")

    checks = {
        "has_layer_sweep_state": "const [layerSweepStep, setLayerSweepStep] = useState(0);" in source,
        "has_layer_sweep_effect": "setLayerSweepStep((prev) => (prev + 1) % LAYER_COUNT);" in source,
        "layer_sweep_waits_for_other_animation": "if (basicRuntimePlaying || predictPlaying || mechanismPlaying)" in source,
        "active_layer_falls_back_to_sweep": ": layerSweepStep;" in source,
        "scene_still_uses_layer_guides": "<LayerGuides activeLayer={activeLayer} />" in source,
    }

    summary = {
        "stage": "stage458_layer_sweep_highlight_restore",
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
