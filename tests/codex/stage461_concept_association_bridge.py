from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage461_concept_association_bridge_{datetime.now().strftime('%Y%m%d')}"


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
        "apple_declares_layer_meta": "const CONCEPT_ASSOCIATION_LAYER_META = [" in apple_source,
        "apple_declares_alias_map": "const CONCEPT_ALIAS_MAP = {" in apple_source,
        "apple_builds_concept_state": "function buildConceptAssociationState(" in apple_source,
        "apple_declares_overlay": "function ConceptAssociationOverlay(" in apple_source,
        "scene_content_accepts_concept_state": "conceptAssociationState = null" in scene_content_signature,
        "scene_accepts_concept_state": "conceptAssociationState = null" in scene_signature,
        "scene_renders_concept_overlay": "<ConceptAssociationOverlay conceptAssociationState={conceptAssociationState} />" in apple_source,
        "scene_content_call_receives_concept_state": "conceptAssociationState={conceptAssociationState}" in scene_content_call,
        "workspace_scene_call_receives_concept_state": "conceptAssociationState={workspace.conceptAssociationState}" in workspace_scene_call,
        "workspace_exports_concept_state": "conceptAssociationState," in apple_source,
        "panel_reads_concept_state": "const conceptAssociationState = workspace?.conceptAssociationState || null;" in panel_source,
        "panel_has_concept_section": "<span>概念关联</span>" in panel_source,
        "panel_mentions_six_layers": "基础编码、静态编码层、动态路径层、结果回收层、传播编码层、语义角色层" in panel_source,
    }

    summary = {
        "stage": "stage461_concept_association_bridge",
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
