from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_TAB_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage442_puzzle_workspace_scene_bridge_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = APPLE_TAB_PATH.read_text(encoding="utf-8")

    checks = {
        "imports_persisted_puzzle_records": "persisted_puzzle_records_v1" in source,
        "has_puzzle_layer_normalizer": "normalizePuzzleResearchLayer" in source,
        "has_puzzle_display_preset": "buildPuzzleDisplayPreset" in source,
        "has_puzzle_selection_matcher": "isNodeMatchedByPuzzle" in source,
        "has_puzzle_selection_candidate": "findPuzzleSelectionCandidate" in source,
        "workspace_tracks_active_puzzle": "activePuzzleRecord" in source and "languageFocus?.activePuzzleId" in source,
        "puzzle_updates_display_levels": "setDisplayLevels((prev) => ({ ...prev, ...preset.displayLevels }))" in source,
        "puzzle_updates_selected_node": "findPuzzleSelectionCandidate(nodes, activePuzzleRecord)" in source and "setSelected(nextSelected)" in source,
        "puzzle_updates_research_layer": "setLanguageFocus((prev) => {" in source and "researchLayer: nextResearchLayer" in source,
    }

    summary = {
        "stage": "stage442_puzzle_workspace_scene_bridge",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "apple_tab": str(APPLE_TAB_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
