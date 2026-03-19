from __future__ import annotations

from stage56_fullsample_regression_outline import build_outline


def test_build_outline_contains_all_feature_families() -> None:
    out = build_outline()
    assert out["record_type"] == "stage56_fullsample_regression_outline_summary"
    assert len(out["feature_families"]) == 5
    assert "β_static" in out["proto_regression_equation"]
