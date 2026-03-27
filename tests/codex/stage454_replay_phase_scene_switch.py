from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage454_replay_phase_scene_switch_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    apple_source = APPLE_PATH.read_text(encoding="utf-8")
    panel_source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "apple_tracks_selected_phase": "selectedRepairReplayPhase: null" in apple_source and "const selectedRepairReplayPhase = languageFocus?.selectedRepairReplayPhase || null;" in apple_source,
        "apple_has_phase_meta_helpers": "function getReplaySlotPhaseMeta" in apple_source and "function getReplayPhaseResearchLayer" in apple_source,
        "apple_phase_enters_slot_focus": "buildRepairReplaySlotFocus(replaySlot, sharedSubcircuitCandidates, replayPhase)" in apple_source,
        "compare_state_accepts_replay_phase": "function buildPuzzleCompareState(nodes = [], links = [], primaryPuzzle = null, comparePuzzle = null, replaySlot = null, replayPhase = null)" in apple_source,
        "apple_phase_changes_display": "mechanism_chain: activePhaseId !== 'before'" in apple_source and "getReplayPhaseResearchLayer(activePhaseId)" in apple_source,
        "scene_overlay_mentions_phase": "阶段 ${puzzleCompareState.replaySlotFocus.activePhaseLabel}" in apple_source,
        "panel_tracks_selected_phase": "selectedRepairReplayPhase: null" in panel_source and "const selectedRepairReplayPhase = languageFocus.selectedRepairReplayPhase || null;" in panel_source,
        "panel_resolves_phase_from_slot": "const resolvedRepairReplayPhase = useMemo(" in panel_source,
        "panel_has_phase_action": "handleSelectRepairReplayPhase" in panel_source and "selectedRepairReplayPhase: phaseId" in panel_source,
        "panel_renders_phase_buttons": "slot.phase_slots.map((phase) => (" in panel_source,
    }

    summary = {
        "stage": "stage454_replay_phase_scene_switch",
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
