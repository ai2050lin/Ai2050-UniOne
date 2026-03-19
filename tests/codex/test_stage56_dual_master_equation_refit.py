from __future__ import annotations

from stage56_dual_master_equation_refit import build_rows


def test_dual_master_equation_refit_builds_general_and_strict_terms() -> None:
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
    assert abs(row["general_balance_v4_term"] - 3.5) < 1e-9
    assert abs(row["kernel_v4_term"] - 3.1) < 1e-9
    assert abs(row["strict_module_term"] - 1.7) < 1e-9
    assert abs(row["dual_gap_term"] - 1.4) < 1e-9
