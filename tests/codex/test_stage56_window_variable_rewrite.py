from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_window_variable_rewrite import (  # noqa: E402
    build_summary,
    contiguous_windows,
    scan_component_windows,
    window_label,
    window_sum,
)


def test_contiguous_windows_covers_expected_ranges() -> None:
    windows = contiguous_windows(length=4, min_width=2, max_width=3)
    assert (0, 2) in windows
    assert (1, 4) in windows
    assert (0, 4) not in windows


def test_window_sum_uses_absolute_values() -> None:
    assert window_sum([1.0, -2.0, 3.0], 0, 2) == 3.0


def test_scan_component_windows_finds_best_window() -> None:
    rows = [
        {
            "tail_position_labels": ["tail_pos_-4", "tail_pos_-3", "tail_pos_-2", "tail_pos_-1"],
            "hidden_token_profile": [0.0, 1.0, 5.0, 0.0],
            "mlp_token_profile": [0.0, 0.0, 1.0, 0.0],
            "union_synergy_joint": 0.1,
            "union_joint_adv": 0.2,
            "weight": 1.0,
        },
        {
            "tail_position_labels": ["tail_pos_-4", "tail_pos_-3", "tail_pos_-2", "tail_pos_-1"],
            "hidden_token_profile": [0.0, 2.0, 8.0, 0.0],
            "mlp_token_profile": [0.0, 0.0, 2.0, 0.0],
            "union_synergy_joint": 0.2,
            "union_joint_adv": 0.3,
            "weight": 1.0,
        },
    ]
    best = scan_component_windows(rows, "hidden_token_profile", "union_synergy_joint", min_width=2, max_width=2)
    assert best["best_window"] == "tail_pos_-3..tail_pos_-2"
    assert best["best_corr"] > 0.0


def test_build_summary_keeps_component_labels() -> None:
    rows = [
        {
            "component_label": "logic_prototype",
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "melon",
            "instance_term": "papaya",
            "weight": 1.0,
            "union_synergy_joint": 0.1,
            "union_joint_adv": 0.2,
            "tail_position_labels": ["tail_pos_-2", "tail_pos_-1"],
            "hidden_token_profile": [0.1, 0.3],
            "mlp_token_profile": [0.2, 0.4],
        },
        {
            "component_label": "logic_fragile_bridge",
            "model_id": "demo",
            "category": "fruit",
            "prototype_term": "kiwi",
            "instance_term": "papaya",
            "weight": 1.0,
            "union_synergy_joint": -0.1,
            "union_joint_adv": -0.2,
            "tail_position_labels": ["tail_pos_-2", "tail_pos_-1"],
            "hidden_token_profile": [0.4, 0.2],
            "mlp_token_profile": [0.3, 0.1],
        },
    ]
    summary = build_summary(rows)
    assert summary["joined_row_count"] == 2
    assert summary["component_labels"] == ["logic_fragile_bridge", "logic_prototype"]
    assert window_label(["tail_pos_-2", "tail_pos_-1"], 0, 2) == "tail_pos_-2..tail_pos_-1"
