from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage449_repair_before_after_contrast_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "has_repair_detector": "isRepairCandidatePuzzle" in source and "repair_candidate" in source,
        "has_repair_summary_builder": "buildRepairContrastSummary" in source and "修复趋势未定" in source,
        "uses_repair_summary_memo": "const repairContrastSummary = useMemo(" in source,
        "has_before_after_section": "反例修复前后对照" in source and "修复前" in source and "修复后" in source,
        "has_signal_contrast_cards": "修复前均值信号" in source and "修复后均值信号" in source and "共享变量增益" in source,
        "has_gain_and_regression_lists": "收益变量" in source and "回退变量" in source,
        "has_validation_reference": "验证参考:" in source and "桥链占比" in source and "共享锚点率" in source,
    }

    summary = {
        "stage": "stage449_repair_before_after_contrast",
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
