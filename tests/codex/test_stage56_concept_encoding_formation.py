from __future__ import annotations

from stage56_concept_encoding_formation import build_concept_encoding_formation_summary


def test_concept_encoding_formation_summary_has_expected_keys() -> None:
    summary = build_concept_encoding_formation_summary()
    hm = summary["headline_metrics"]

    assert hm["family_anchor_strength"] > 0.0
    assert hm["concept_seed_drive"] > hm["concept_binding_drive"] > 0.0
    assert hm["concept_embedding_drive"] > 0.0
    assert hm["concept_encoding_margin"] > 0.0
    assert hm["apple_banana_transfer_support"] > 0.5


def test_apple_top_fibers_contains_round_or_sweet() -> None:
    summary = build_concept_encoding_formation_summary()
    attrs = [item["attribute"] for item in summary["apple_top_fibers"][:3]]

    assert "round" in attrs or "sweet" in attrs


def test_fruit_chart_reconstruction_is_reasonable() -> None:
    summary = build_concept_encoding_formation_summary()
    hm = summary["headline_metrics"]

    assert hm["fruit_chart_reconstruction_error_mean"] < hm["apple_local_offset_norm"] * 2.0
