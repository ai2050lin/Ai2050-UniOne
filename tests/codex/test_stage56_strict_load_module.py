from __future__ import annotations

from stage56_strict_load_module import build_rows


def test_strict_load_module_builds_four_variants() -> None:
    rows = build_rows(
        [
            {
                "strict_load_term": 2.0,
                "logic_strictload_term": 0.5,
            }
        ]
    )
    row = rows[0]
    assert abs(row["strict_module_base_term"] - 2.0) < 1e-9
    assert abs(row["strict_module_logic_term"] - 0.5) < 1e-9
    assert abs(row["strict_module_combined_term"] - 2.5) < 1e-9
    assert abs(row["strict_module_residual_term"] - 1.5) < 1e-9
