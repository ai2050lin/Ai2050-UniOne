from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_FILE = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
APP_FILE = ROOT / "frontend" / "src" / "App.jsx"
OUT_DIR = ROOT / "tests" / "codex_temp" / "stage439_legacy_control_panels_retire_20260326"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    apple_text = APPLE_FILE.read_text(encoding="utf-8")
    app_text = APP_FILE.read_text(encoding="utf-8")

    summary = {
        "stage": "stage439_legacy_control_panels_retire",
        "apple_control_panels_export_removed": "export function AppleNeuronControlPanels" not in apple_text,
        "legacy_comment_present": "legacy control panels removed from main path" in apple_text,
        "app_direct_usage_removed": "AppleNeuronControlPanels" not in app_text,
    }
    summary["all_passed"] = all(
        summary[key]
        for key in (
            "apple_control_panels_export_removed",
            "legacy_comment_present",
            "app_direct_usage_removed",
        )
    )

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
