from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROADMAP_PATH = ROOT / "frontend" / "src" / "blueprint" / "ProjectRoadmapTab.jsx"
DOC_PATH = ROOT / "research" / "gpt5" / "docs" / "AGI_GPT5_INTERFACE_MODULE_DRAFT.md"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage466_interface_module_draft_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    roadmap_source = ROADMAP_PATH.read_text(encoding="utf-8")
    doc_source = DOC_PATH.read_text(encoding="utf-8")

    checks = {
        "doc_exists": DOC_PATH.exists(),
        "doc_has_title": "AGI_GPT5 界面模块重组草稿 V1" in doc_source,
        "doc_has_wireframe": "主界面草稿图" in doc_source and "左侧操作栏" in doc_source and "右侧数据面板" in doc_source,
        "doc_has_rules": "左侧只放操作" in doc_source and "路线图只放战略与理论" in doc_source,
        "roadmap_has_interface_draft_section": "界面模块草稿图" in roadmap_source,
        "roadmap_has_wireframe_constant": "INTERFACE_MODULE_DRAFT_WIREFRAME" in roadmap_source,
        "roadmap_mentions_five_roles": "操作、观察、数据、验证、战略" in roadmap_source,
    }

    summary = {
        "stage": "stage466_interface_module_draft",
        "all_passed": all(checks.values()),
        "checks": checks,
        "files": {
            "roadmap": str(ROADMAP_PATH.relative_to(ROOT)).replace("\\", "/"),
            "doc": str(DOC_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }

    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
