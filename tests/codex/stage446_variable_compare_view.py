from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage446_variable_compare_view_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "has_variable_meta": "const VARIABLE_META" in source and "实体锚点" in source and "闭合读出" in source,
        "has_variable_role_map": "const VARIABLE_ROLE_MAP" in source and "hardBinding" in source,
        "uses_workspace_nodes": "const workspaceNodes = Array.isArray(workspace?.nodes) ? workspace.nodes : [];" in source,
        "has_variable_stat_builder": "buildVariableNodeStat" in source and "getNodeSignal" in source,
        "has_variable_compare_builder": "buildVariableCompareRows" in source,
        "has_variable_compare_section": "变量级对比视图" in source and "分析节点池" in source,
        "shows_variable_signal_summary": "主均值信号" in source and "对比均值信号" in source,
        "shows_variable_peak_summary": "主峰值层" in source and "对比峰值层" in source,
    }

    summary = {
        "stage": "stage446_variable_compare_view",
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
