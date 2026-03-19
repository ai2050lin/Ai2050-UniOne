from __future__ import annotations

from stage56_static_raw_chain import build_static_raw_rows, build_summary


def test_build_static_raw_rows_adds_category_relative_proxies() -> None:
    rows = [
        {"model_id": "m", "category": "fruit", "atlas_static_proxy": 0.6, "offset_static_proxy": 0.2},
        {"model_id": "m", "category": "fruit", "atlas_static_proxy": 0.4, "offset_static_proxy": 0.4},
    ]
    out = build_static_raw_rows(rows)
    assert len(out) == 2
    assert "atlas_raw_proxy" in out[0]
    assert "offset_raw_proxy" in out[0]
    summary = build_summary(out)
    assert summary["record_type"] == "stage56_static_raw_chain_summary"
