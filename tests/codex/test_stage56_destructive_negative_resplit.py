from __future__ import annotations

from stage56_destructive_negative_resplit import build_rows


def test_resplit_separates_core_and_strict_load() -> None:
    rows = build_rows(
        [
            {
                "logic_fragile_bridge_term": 1.2,
                "frontier_negative_base_term": 0.3,
                "window_gate_negative_term": 0.5,
                "alignment_load_v2_term": -0.2,
            }
        ]
    )
    row = rows[0]
    assert abs(row["destructive_core_term"] - 1.2) < 1e-9
    assert abs(row["strict_load_term"] - 0.8) < 1e-9
    assert abs(row["destructive_alignment_term"] - 1.0) < 1e-9
