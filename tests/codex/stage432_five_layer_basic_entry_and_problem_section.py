from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "frontend" / "src" / "components" / "LanguageResearchControlPanel.jsx"
OUT_DIR = ROOT / "tests" / "codex_temp" / "stage432_five_layer_basic_entry_and_problem_section_20260326"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    checks = {
        "basic_button_opens_window": "onClick={() => setLegacyOpen(true)}" in text,
        "five_layer_section_exists": "RESEARCH_LAYERS.map((item) => (" in text,
        "basic_problem_section_exists": "HARD_PROBLEMS.map((item) => {" in text and "ShieldAlert" in text,
    }
    summary = {
        "stage": "stage432_five_layer_basic_entry_and_problem_section",
        "all_passed": all(checks.values()),
        "checks": checks,
        "target": str(TARGET),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
