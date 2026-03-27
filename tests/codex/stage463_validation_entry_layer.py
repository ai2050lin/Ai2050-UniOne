from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage463_validation_entry_layer_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "declares_validation_builder": "function buildValidationEntrySummary(" in source,
        "builds_validation_summary": "const validationEntrySummary = useMemo(" in source,
        "renders_validation_section": "<span>验证入口</span>" in source,
        "validation_mentions_real_samples": "真实样本验证现在走到哪一步" in source,
        "validation_mentions_priority_gap": "优先缺口" in source,
        "validation_mentions_phase_coverage": "阶段覆盖" in source,
        "validation_mentions_slot_status": "槽位状态" in source,
        "validation_mentions_current_focus": "当前聚焦" in source,
        "validation_mentions_validation_judge": "当前验证判断" in source,
        "validation_uses_priority_asset": "nextPriorityAsset" in source,
        "validation_uses_phase_counts": "beforeCoverage" in source and "bridgeCoverage" in source and "afterCoverage" in source,
    }

    summary = {
        "stage": "stage463_validation_entry_layer",
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
