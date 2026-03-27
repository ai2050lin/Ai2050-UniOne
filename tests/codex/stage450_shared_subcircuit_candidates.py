from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage450_shared_subcircuit_candidates_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    apple_source = APPLE_PATH.read_text(encoding="utf-8")
    panel_source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "compare_state_tracks_shared_variables": "sharedVariables" in apple_source,
        "compare_state_builds_shared_subcircuits": "sharedSubcircuitCandidates" in apple_source and "shared-subcircuit-" in apple_source,
        "candidate_has_score_and_reason": "candidate.score" in panel_source and "candidate.reason" in panel_source,
        "workspace_reads_shared_subcircuits": "const sharedSubcircuitCandidates = workspace?.puzzleCompareState?.sharedSubcircuitCandidates || [];" in panel_source,
        "panel_has_shared_subcircuit_section": "最小共享子回路候选" in panel_source,
        "panel_has_shared_subcircuit_summary": "候选数" in panel_source and "局部回放链" in panel_source,
        "panel_has_candidate_detail_fields": "命中共享变量" in panel_source and "共享端点数" in panel_source and "候选类型" in panel_source,
    }

    summary = {
        "stage": "stage450_shared_subcircuit_candidates",
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
