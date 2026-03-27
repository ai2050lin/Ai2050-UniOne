from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage448_local_replay_validation_view_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    apple_source = APPLE_PATH.read_text(encoding="utf-8")
    panel_source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "compare_state_has_validation_metrics": "minimalityScore" in apple_source and "sharedAnchorRate" in apple_source and "bridgeDominance" in apple_source,
        "compare_state_has_validation_label": "validationLabel" in apple_source and "裁剪较稳" in apple_source,
        "scene_overlay_shows_validation": "裁剪验证" in apple_source and "最小性" in apple_source,
        "panel_reads_compare_validation": "const compareValidation = workspace?.puzzleCompareState?.validation || null;" in panel_source,
        "panel_has_validation_section": "局部链路裁剪验证" in panel_source and "最小性分数" in panel_source,
        "panel_shows_validation_breakdown": "桥链占比" in panel_source and "共享锚点率" in panel_source and "平均跨层" in panel_source,
    }

    summary = {
        "stage": "stage448_local_replay_validation_view",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "apple_tab": str(APPLE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "panel": str(PANEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
