from __future__ import annotations

from stage56_closed_form_v4_refit import build_rows


def test_closed_form_v4_refit_builds_short_kernel() -> None:
    rows = build_rows(
        [
            {
                "core_balance_v3_term": 3.0,
                "logic_strictload_term": 0.5,
                "style_structure_gain_term": -0.4,
            }
        ]
    )
    row = rows[0]
    assert abs(row["style_penalty_term"] - 0.4) < 1e-9
    assert abs(row["general_balance_v4_term"] - 3.5) < 1e-9
    assert abs(row["kernel_v4_term"] - 3.1) < 1e-9
