from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage453_replay_slot_scene_focus_bridge_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    apple_source = APPLE_PATH.read_text(encoding="utf-8")
    panel_source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "apple_imports_replay_slots": "PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1" in apple_source,
        "default_focus_tracks_selected_slot": "selectedRepairReplaySlotId: null" in apple_source,
        "apple_builds_slot_focus": "function buildRepairReplaySlotFocus" in apple_source and "replaySlotFocus" in apple_source,
        "compare_state_accepts_replay_slot": "function buildPuzzleCompareState(nodes = [], links = [], primaryPuzzle = null, comparePuzzle = null, replaySlot = null, replayPhase = null)" in apple_source,
        "compare_state_exports_scene_maps": "sceneLinkHighlightMap" in apple_source and "sceneNodeCategoryMap" in apple_source and "sceneNodeHighlightMap" in apple_source,
        "workspace_reads_selected_slot": "const selectedRepairReplaySlot = useMemo(" in apple_source,
        "workspace_applies_slot_focus_effect": "lastAppliedReplaySlotIdRef" in apple_source and "setShowAlgorithmRuntimeChain(true);" in apple_source,
        "scene_reads_scene_focus_maps": "sceneLinkHighlightMap || puzzleCompareState?.linkHighlightMap" in apple_source or "puzzleCompareState?.sceneLinkHighlightMap || puzzleCompareState?.linkHighlightMap" in apple_source,
        "scene_overlay_mentions_slot_focus": "回放槽位" in apple_source and "聚焦链路" in apple_source,
        "panel_tracks_selected_slot": "selectedRepairReplaySlotId" in panel_source and "const selectedRepairReplaySlot = useMemo(" in panel_source,
        "panel_has_slot_focus_actions": "handleSelectRepairReplaySlot" in panel_source and "投到场景" in panel_source and "取消聚焦" in panel_source,
    }

    summary = {
        "stage": "stage453_replay_slot_scene_focus_bridge",
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
