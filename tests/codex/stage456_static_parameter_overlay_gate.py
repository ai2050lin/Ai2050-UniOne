from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage456_static_parameter_overlay_gate_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = APPLE_PATH.read_text(encoding="utf-8")

    checks = {
        "has_overlay_gate_flag": "const shouldRenderParameterStateOverlay = Boolean(" in source,
        "gate_checks_parameter_level": "displayLevels?.parameter_state !== false" in source,
        "gate_blocks_default_static_layer": "runtimeLayerKey !== 'static_encoding'" in source,
        "gate_allows_manual_static_encoding": "showAlgorithmStaticEncoding" in source,
        "gate_allows_replay_slot_focus": "languageFocus?.selectedRepairReplaySlotId" in source,
        "overlay_uses_gate": "{shouldRenderParameterStateOverlay ? (" in source,
    }

    summary = {
        "stage": "stage456_static_parameter_overlay_gate",
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
