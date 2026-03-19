from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_natural_generation_window_probe import natural_prompt_variants, write_report  # noqa: E402


def test_natural_prompt_variants_cover_all_axes_once() -> None:
    rows = natural_prompt_variants("papaya")
    assert [row["axis"] for row in rows] == ["control", "style", "logic", "syntax"]
    assert all("papaya" in row["prompt"] for row in rows)


def test_write_report_includes_generated_token_count(tmp_path: Path) -> None:
    summary = {
        "case_count": 1,
        "model_count": 1,
        "tail_tokens": 16,
        "per_model": {
            "demo/model": {
                "case_count": 1,
                "per_axis": {
                    "logic": {
                        "dominant_hidden_tail_position": "tail_pos_-5",
                        "dominant_mlp_tail_position": "tail_pos_-4",
                        "dominant_hidden_layer": "layer_10",
                        "dominant_mlp_layer": "layer_11",
                    }
                },
            }
        },
    }
    rows = [
        {
            "model_id": "demo/model",
            "category": "fruit",
            "axis": "logic",
            "instance_term": "papaya",
            "generated_token_count": 6,
            "generated_text": "fruit.",
        }
    ]
    report_path = tmp_path / "REPORT.md"
    write_report(report_path, summary, rows)
    text = report_path.read_text(encoding="utf-8")
    assert "tokens=6" in text
    assert "fruit." in text
