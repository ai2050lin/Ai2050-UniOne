from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage445_puzzle_compare_panel_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "has_compare_focus_state": "comparePuzzleId" in source and "handleComparePuzzleSelect" in source,
        "has_compare_section": "拼图对比台" in source and "清空对比" in source,
        "has_layer_relation_summary": "buildLayerRelation" in source and "层关系" in source,
        "has_variable_overlap_summary": "共同变量" in source and "主拼图独有变量" in source and "对比拼图独有变量" in source,
        "has_compare_action_summary": "主下一步" in source and "对比下一步" in source,
        "supports_explicit_clear_compare": "hasOwnProperty.call(languageFocus || {}, 'comparePuzzleId')" in source,
    }

    summary = {
        "stage": "stage445_puzzle_compare_panel",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "panel": str(PANEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
