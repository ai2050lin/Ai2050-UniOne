from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage457_default_object_family_hidden_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = APPLE_PATH.read_text(encoding="utf-8")

    checks = {
        "basic_preset_hides_object_family": "basic_only" in source and "object_family: false" in source,
        "display_levels_default_hide_object_family": "const [displayLevels, setDisplayLevels] = useState({\n    basic_neurons: true,\n    object_family: false," in source,
        "mount_effect_hides_object_family": "basic_neurons: true,\n      object_family: false,\n      parameter_state: false," in source,
        "nodes_gate_object_family": "const objectFamilyVisible = displayLevels?.object_family !== false;" in source,
        "selected_defaults_to_null": "const [selected, setSelected] = useState(null);" in source,
        "selected_falls_back_to_visible_node": "const fallbackVisibleNode = nodes.find((node) => node.role !== 'background') || null;" in source,
    }

    summary = {
        "stage": "stage457_default_object_family_hidden",
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
