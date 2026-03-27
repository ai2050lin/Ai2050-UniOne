from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage459_layer_sweep_scene_contract_{datetime.now().strftime('%Y%m%d')}"


def extract_block(source: str, pattern: str) -> str:
    match = re.search(pattern, source, flags=re.S)
    return match.group(1) if match else ""


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = APPLE_PATH.read_text(encoding="utf-8")

    scene_content_signature = extract_block(
        source,
        r"export function AppleNeuronSceneContent\(\{(.*?)\}\)\s*\{",
    )
    scene_signature = extract_block(
        source,
        r"function AppleNeuronScene\(\{(.*?)\}\)\s*\{",
    )
    scene_content_call = extract_block(
        source,
        r"<AppleNeuronSceneContent\s+(.*?)\/>",
    )
    workspace_scene_call = extract_block(
        source,
        r"<AppleNeuronScene\s+(.*?)\/>",
    )

    checks = {
        "workspace_tracks_layer_sweep_state": "const [layerSweepStep, setLayerSweepStep] = useState(0);" in source,
        "workspace_exports_layer_sweep": "layerSweepStep," in source,
        "scene_content_signature_accepts_layer_sweep": "layerSweepStep = 0" in scene_content_signature,
        "scene_signature_accepts_layer_sweep": "layerSweepStep = 0" in scene_signature,
        "scene_content_call_receives_layer_sweep": "layerSweepStep={layerSweepStep}" in scene_content_call,
        "workspace_scene_call_receives_layer_sweep": "layerSweepStep={workspace.layerSweepStep}" in workspace_scene_call,
        "scene_uses_layer_sweep": ": layerSweepStep;" in source,
    }

    summary = {
        "stage": "stage459_layer_sweep_scene_contract",
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
