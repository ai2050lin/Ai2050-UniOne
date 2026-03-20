from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_icspb_object_remapping import build_icspb_object_remapping_summary


def test_icspb_object_remapping_positive() -> None:
    summary = build_icspb_object_remapping_summary()
    hm = summary["headline_metrics"]

    assert hm["family_patch_to_structure"] > hm["concept_offset_to_feature"]
    assert hm["attribute_fiber_to_feature"] > 0.0
    assert hm["relation_context_to_transport"] > 0.0
    assert 0.0 < hm["remap_consistency"] < 1.0
