from __future__ import annotations

from stage56_dual_master_equation_finalizer import build_rows


def test_dual_master_equation_finalizer_builds_final_terms() -> None:
    rows = build_rows(
        [
            {
                "core_balance_v3_term": 3.0,
                "logic_strictload_term": 0.5,
                "style_structure_gain_term": -0.4,
                "strict_load_term": 1.2,
            }
        ]
    )
    row = rows[0]
    assert abs(row["kernel_v4_term"] - 3.1) < 1e-9
    assert abs(row["strict_module_final_term"] - 1.2) < 1e-9
    assert abs(row["dual_gap_final_term"] - 1.9) < 1e-9
