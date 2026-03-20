from __future__ import annotations

try:
    from tests.codex.stage56_encoding_kernel_cross_asset_validation import (
        build_encoding_kernel_cross_asset_validation_summary,
    )
except ModuleNotFoundError:
    from stage56_encoding_kernel_cross_asset_validation import build_encoding_kernel_cross_asset_validation_summary


def test_cross_asset_support_bounds() -> None:
    summary = build_encoding_kernel_cross_asset_validation_summary()
    hm = summary["headline_metrics"]
    assert 0.0 <= hm["cross_asset_support"] <= 1.0
    assert hm["support_gap"] >= 0.0
    assert hm["formula_support"] > 0.0
