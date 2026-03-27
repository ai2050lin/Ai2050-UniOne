from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage462_overview_property_layers_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    source = PANEL_PATH.read_text(encoding="utf-8")

    checks = {
        "declares_overview_builder": "function buildResearchOverviewSummary(" in source,
        "declares_property_builder": "function buildPropertySummaryCards(" in source,
        "reads_workspace_summary": "const workspaceSummary = workspace?.summary || null;" in source,
        "builds_overview_summary": "const overviewSummary = useMemo(" in source,
        "builds_property_summary_cards": "const propertySummaryCards = useMemo(" in source,
        "renders_overview_section": "<span>研究总览</span>" in source,
        "renders_property_section": "<span>关键性质</span>" in source,
        "overview_mentions_risk": "当前风险" in source,
        "overview_mentions_validation": "验证状态" in source,
        "property_mentions_four_estimates": "当前数据最核心的四个性质估计" in source,
        "property_includes_stability": "label: '稳定性'" in source,
        "property_includes_separability": "label: '可分性'" in source,
        "property_includes_coupling": "label: '耦合强度'" in source,
        "property_includes_completeness": "label: '观测完整度'" in source,
    }

    summary = {
        "stage": "stage462_overview_property_layers",
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
