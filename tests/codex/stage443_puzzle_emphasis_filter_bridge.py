from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPLE_TAB_PATH = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage443_puzzle_emphasis_filter_bridge_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = APPLE_TAB_PATH.read_text(encoding="utf-8")

    checks = {
      "has_variable_role_mapper": "getPuzzleVariablePreferredRoles" in source,
      "has_puzzle_emphasis_builder": "buildPuzzleNodeEmphasisMap" in source,
      "workspace_builds_puzzle_emphasis": "const puzzleNodeEmphasis = useMemo(" in source,
      "node_emphasis_consumes_puzzle_emphasis": "puzzleNodeEmphasis?.[node.id]" in source,
      "selected_node_still_preserved": "if (selected?.id === node.id)" in source,
      "puzzle_emphasis_depends_on_mapped_variables": "puzzleRecord.mappedVariables" in source,
    }

    summary = {
        "stage": "stage443_puzzle_emphasis_filter_bridge",
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
