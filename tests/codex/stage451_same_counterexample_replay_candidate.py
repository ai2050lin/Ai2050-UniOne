from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage451_same_counterexample_replay_candidate_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "has_replay_summary_builder": "buildRepairReplaySummary" in source and "建议先回放共享候选链" in source,
        "uses_replay_summary_memo": "const repairReplaySummary = useMemo(" in source,
        "has_replay_section": "同一反例样本回放候选" in source,
        "has_replay_summary_cards": "锚定变量" in source and "共享候选链" in source and "当前建议" in source,
        "has_three_phase_replay": "修复前" in source and "共享候选链" in source and "修复后" in source,
        "has_replay_verdict": "repairReplaySummary.verdict" in source,
    }

    summary = {
        "stage": "stage451_same_counterexample_replay_candidate",
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
