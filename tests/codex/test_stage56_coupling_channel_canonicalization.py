from __future__ import annotations

from stage56_coupling_channel_canonicalization import build_rows


def test_coupling_channel_canonicalization_flips_load_terms() -> None:
    rows = build_rows(
        [
            {
                "gs_coupling_term": -3.0,
                "gd_coupling_term": 5.0,
                "sd_coupling_term": -2.0,
            }
        ]
    )
    row = rows[0]
    assert abs(row["gs_load_channel_term"] - 3.0) < 1e-9
    assert abs(row["gd_drive_channel_term"] - 5.0) < 1e-9
    assert abs(row["sd_load_channel_term"] - 2.0) < 1e-9
