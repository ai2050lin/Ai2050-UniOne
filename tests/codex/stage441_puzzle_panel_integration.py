from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
PUZZLE_JS_PATH = ROOT / "frontend" / "src" / "blueprint" / "data" / "persisted_puzzle_records_v1.js"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage441_puzzle_panel_integration_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    panel_source = PANEL_PATH.read_text(encoding="utf-8")
    puzzle_source = PUZZLE_JS_PATH.read_text(encoding="utf-8")

    checks = {
        "puzzle_data_file_exists": PUZZLE_JS_PATH.exists(),
        "panel_imports_puzzle_data": "persisted_puzzle_records_v1" in panel_source,
        "panel_has_puzzle_section": "基础拼图仓" in panel_source,
        "panel_has_axis_filter": "priorityAxisCounts" in panel_source and "全部" in panel_source,
        "panel_has_puzzle_selection": "handlePuzzleSelect" in panel_source and "activePuzzleId" in panel_source,
        "panel_shows_next_action": "下一步动作" in panel_source,
        "persisted_puzzle_summary_exported": "PERSISTED_PUZZLE_SUMMARY_V1" in puzzle_source,
        "persisted_puzzle_records_exported": "PERSISTED_PUZZLE_RECORDS_V1" in puzzle_source,
    }

    summary = {
        "stage": "stage441_puzzle_panel_integration",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "panel": str(PANEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "puzzle_data": str(PUZZLE_JS_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
