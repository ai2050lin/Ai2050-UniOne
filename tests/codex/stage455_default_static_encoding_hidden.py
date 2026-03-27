from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage455_default_static_encoding_hidden_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = APPLE_PATH.read_text(encoding="utf-8")

    checks = {
        "has_display_preset_builder": "function buildPuzzleDisplayPreset" in source,
        "static_encoding_returns_base": "case 'static_encoding':\n      return base;" in source,
        "base_keeps_parameter_state_hidden": "parameter_state: false" in source,
        "base_keeps_mechanism_chain_hidden": "mechanism_chain: false" in source,
        "base_keeps_static_encoding_hidden": "showAlgorithmStaticEncoding: false" in source,
        "ui_copy_still_mentions_manual_open": "默认先看基础信息" in source and "需要手动切到“算法显示”后再打开" in source,
    }

    summary = {
        "stage": "stage455_default_static_encoding_hidden",
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
