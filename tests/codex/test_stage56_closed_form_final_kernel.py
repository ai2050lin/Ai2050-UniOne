from __future__ import annotations

from stage56_closed_form_final_kernel import build_rows


def test_closed_form_final_kernel_builds_shorter_terms() -> None:
    rows = build_rows(
        [
            {
                "positive_mass_v2_term": 3.0,
                "destructive_core_term": 1.0,
                "alignment_load_v2_term": -0.5,
                "style_structure_gain_term": 0.4,
                "strict_load_term": 0.2,
            }
        ]
    )
    row = rows[0]
    assert abs(row["core_balance_v3_term"] - 2.5) < 1e-9
    assert abs(row["closed_form_kernel_v3_term"] - 2.9) < 1e-9
    assert abs(row["strict_kernel_v3_term"] - 3.1) < 1e-9
