from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_mass_term_scan_compare import (  # noqa: E402
    build_consensus_rows,
    build_model_row,
    build_summary,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_scan_dir(root: Path, name: str, model_id: str, action_term: str) -> Path:
    scan_dir = root / name
    write_json(
        scan_dir / "manifest.json",
        {
            "model_id": model_id,
            "counts": {"input_items": 24},
        },
    )
    write_json(
        scan_dir / "summary.json",
        {
            "headline_metrics": {
                "record_count": 48,
                "family_count": 24,
                "closure_candidate_count": 24,
                "mean_prompt_stability_survey": 0.7,
                "mean_prompt_stability_deep": 0.5,
                "mean_prompt_stability_closure": 0.4,
            },
            "category_coverage_survey": {"action": 2, "weather": 2},
        },
    )
    write_jsonl(
        scan_dir / "closure_candidates.jsonl",
        [
            {
                "pool": "closure",
                "item": {"term": action_term, "category": "action"},
                "exact_closure_proxy": 0.61,
                "wrong_family_margin": 0.12,
            },
            {
                "pool": "closure",
                "item": {"term": "humidity", "category": "weather"},
                "exact_closure_proxy": 0.63,
                "wrong_family_margin": 0.11,
            },
        ],
    )
    return scan_dir


def test_compare_builds_model_rows_and_consensus(tmp_path):
    left = make_scan_dir(tmp_path, "deepseek_large", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "open")
    right = make_scan_dir(tmp_path, "qwen_large", "Qwen/Qwen3-4B", "open")

    left_row = build_model_row(left)
    assert left_row["input_items"] == 24
    assert left_row["top_terms_by_category"][0]["category"] == "action"

    consensus_rows = build_consensus_rows([left, right])
    action_row = next(row for row in consensus_rows if row["category"] == "action")
    assert action_row["consensus_term"] == "open"
    assert action_row["consensus_support_count"] == 2

    summary = build_summary([left, right], [build_model_row(left), build_model_row(right)], consensus_rows)
    assert summary["model_count"] == 2
    assert "action" in summary["strong_consensus_categories"]
