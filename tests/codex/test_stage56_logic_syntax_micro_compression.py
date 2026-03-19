from __future__ import annotations

from stage56_logic_syntax_micro_compression import build_rows


def test_logic_syntax_micro_compression_builds_support_and_interference() -> None:
    rows = build_rows(
        [
            {
                "positive_mass_v2_term": 3.0,
                "destructive_core_term": 1.0,
                "alignment_load_v2_term": -0.5,
                "strict_load_term": 0.4,
                "logic_structure_gain_term": 2.0,
                "syntax_structure_gain_term": 3.0,
            }
        ]
    )
    row = rows[0]
    assert abs(row["core_balance_v3_term"] - 2.5) < 1e-9
    assert abs(row["logic_core_term"] - 5.0) < 1e-9
    assert abs(row["logic_strictload_term"] - 0.8) < 1e-9
    assert abs(row["syntax_core_term"] - 7.5) < 1e-9
    assert abs(row["syntax_strictload_term"] - 1.2) < 1e-9
    assert abs(row["logic_syntax_support_term"] - 14.5) < 1e-9
    assert abs(row["logic_syntax_interference_term"] - 6.0) < 1e-9
    assert abs(row["logic_net_support_term"] - 4.2) < 1e-9
    assert abs(row["syntax_net_support_term"] + 6.3) < 1e-9
    assert abs(row["logic_syntax_net_support_term"] + 2.1) < 1e-9
