from __future__ import annotations

from stage56_family_patch_offset_raw_chain import build_rows, build_summary


def test_build_rows_adds_family_patch_and_concept_offset() -> None:
    design_rows = [
        {"model_id": "m", "category": "fruit", "atlas_static_proxy": 0.6, "offset_static_proxy": 0.2, "frontier_dynamic_proxy": 0.5},
        {"model_id": "m", "category": "fruit", "atlas_static_proxy": 0.4, "offset_static_proxy": 0.4, "frontier_dynamic_proxy": 0.7},
    ]
    rows = build_rows(design_rows)
    assert len(rows) == 2
    assert "family_patch_raw" in rows[0]
    assert "concept_offset_raw" in rows[0]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_family_patch_offset_raw_chain_summary"
