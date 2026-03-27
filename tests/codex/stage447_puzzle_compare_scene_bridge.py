from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage447_puzzle_compare_scene_bridge_{datetime.now().strftime('%Y%m%d')}"


def extract_block(source: str, pattern: str) -> str:
    match = re.search(pattern, source, flags=re.S)
    return match.group(1) if match else ""


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    apple_source = APPLE_PATH.read_text(encoding="utf-8")
    panel_source = PANEL_PATH.read_text(encoding="utf-8")

    scene_content_signature = extract_block(
        apple_source,
        r"export function AppleNeuronSceneContent\(\{(.*?)\}\)\s*\{",
    )
    scene_signature = extract_block(
        apple_source,
        r"function AppleNeuronScene\(\{(.*?)\}\)\s*\{",
    )
    scene_content_call = extract_block(
        apple_source,
        r"<AppleNeuronSceneContent\s+(.*?)\/>",
    )
    workspace_scene_call = extract_block(
        apple_source,
        r"<AppleNeuronScene\s+(.*?)\/>",
    )

    checks = {
        "has_compare_state_builder": "buildPuzzleCompareState" in apple_source and "buildPuzzleFocusNodeIdSet" in apple_source,
        "workspace_tracks_compare_puzzle": "const comparePuzzleRecord = useMemo(" in apple_source,
        "workspace_exports_compare_state": "puzzleCompareState," in apple_source,
        "scene_content_signature_accepts_compare_state": "puzzleCompareState = null" in scene_content_signature,
        "scene_signature_accepts_compare_state": "puzzleCompareState = null" in scene_signature,
        "scene_content_call_receives_compare_state": "puzzleCompareState={puzzleCompareState}" in scene_content_call,
        "workspace_scene_call_receives_compare_state": "puzzleCompareState={workspace.puzzleCompareState}" in workspace_scene_call,
        "workspace_scene_call_receives_language_focus": "languageFocus={workspace.languageFocus}" in workspace_scene_call,
        "scene_renders_compare_links": "puzzleCompareVisibleLinks" in apple_source and "puzzle-compare-" in apple_source,
        "scene_renders_compare_nodes": "puzzleCompareVisibleNodes" in apple_source and "puzzle-compare-node-" in apple_source,
        "scene_shows_compare_overlay": "双拼图差异高亮" in apple_source and "局部链路回放" in apple_source,
        "panel_reads_scene_compare_summary": "compareSceneSummary" in panel_source and "场景差异摘要" in panel_source,
    }

    summary = {
        "stage": "stage447_puzzle_compare_scene_bridge",
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
