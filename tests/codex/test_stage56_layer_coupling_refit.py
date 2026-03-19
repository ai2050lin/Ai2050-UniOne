from __future__ import annotations

from stage56_layer_coupling_refit import build_rows


def test_layer_coupling_refit_builds_three_couplings() -> None:
    rows = build_rows(
        [
            {
                "style_structure_gain_term": -0.4,
                "core_balance_v3_term": 3.0,
                "logic_strictload_term": 0.5,
                "strict_load_term": 1.2,
            }
        ]
    )
    row = rows[0]
    assert abs(row["kernel_v4_term"] - 3.1) < 1e-9
    assert abs(row["strict_module_final_term"] - 1.2) < 1e-9
    assert abs(row["dual_gap_final_term"] - 1.9) < 1e-9
    assert abs(row["gs_coupling_term"] - 3.72) < 1e-9
    assert abs(row["gd_coupling_term"] - 5.89) < 1e-9
    assert abs(row["sd_coupling_term"] - 2.28) < 1e-9
