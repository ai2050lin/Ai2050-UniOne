from __future__ import annotations

try:
    from tests.codex.stage56_apple_banana_encoding_transfer import build_apple_banana_encoding_transfer_summary
except ModuleNotFoundError:
    from stage56_apple_banana_encoding_transfer import build_apple_banana_encoding_transfer_summary


def test_apple_banana_transfer_prefers_banana_to_cat() -> None:
    summary = build_apple_banana_encoding_transfer_summary()
    hm = summary["headline_metrics"]
    assert hm["pred_vs_banana_cosine"] > hm["pred_vs_cat_cosine"]
    assert hm["banana_language_cosine"] > 0.5
    assert hm["predicted_elongated_alignment"] > hm["predicted_round_alignment"]
