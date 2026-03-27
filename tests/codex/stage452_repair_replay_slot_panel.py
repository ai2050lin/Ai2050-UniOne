from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage452_repair_replay_slot_panel_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "imports_replay_slot_data": "PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1" in source and "PERSISTED_REPAIR_REPLAY_SLOT_META_V1" in source,
        "has_status_label_map": "REPAIR_REPLAY_STATUS_LABEL_MAP" in source and "statusLegend" in source,
        "builds_replay_slot_summary": "function buildRepairReplaySlotSummary" in source and "matchTypeLabel" in source,
        "uses_replay_slot_summary_memo": "const repairReplaySlotSummary = useMemo(" in source,
        "panel_has_replay_slot_section": "真实样本回放槽位" in source and "已匹配槽位" in source and "平均就绪度" in source,
        "panel_has_slot_detail_fields": "共享候选链" in source and "缺失资产" in source and "阶段:" in source,
        "panel_has_exact_vs_repair_only": "精确匹配" in source and "修复侧候选" in source,
    }

    summary = {
      "stage": "stage452_repair_replay_slot_panel",
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
