from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_bxm_rewrite import build_summary, rewrite_rows  # noqa: E402


def make_row(union_joint_adv: float, union_synergy_joint: float, bridge: float, conflict: float, mismatch: float) -> dict:
    return {
        "union_joint_adv": union_joint_adv,
        "union_synergy_joint": union_synergy_joint,
        "strict_positive_synergy": union_joint_adv > 0 and union_synergy_joint > 0,
        "axes": {
            axis: {
                "bridge_field_proxy": bridge,
                "conflict_field_proxy": conflict,
                "mismatch_field_proxy": mismatch,
            }
            for axis in ("style", "logic", "syntax")
        },
    }


def test_rewrite_rows_splits_supportive_and_destructive_components() -> None:
    rows = rewrite_rows(
        [
            make_row(0.2, 0.1, 0.4, 0.3, 0.5),
            make_row(0.1, -0.2, 0.4, 0.3, 0.5),
        ]
    )
    first_logic = rows[0]["rewritten_axes"]["logic"]
    second_logic = rows[1]["rewritten_axes"]["logic"]
    assert first_logic["stable_bridge"] == 0.4
    assert first_logic["fragile_bridge"] == 0.0
    assert second_logic["stable_bridge"] == 0.0
    assert second_logic["fragile_bridge"] == 0.4
    assert first_logic["constraint_conflict"] == 0.3
    assert second_logic["destructive_conflict"] == 0.3
    assert first_logic["mismatch_exposure"] == 0.5
    assert second_logic["mismatch_damage"] == 0.5


def test_build_summary_reports_component_shares() -> None:
    rows = rewrite_rows(
        [
            make_row(0.2, 0.1, 0.6, 0.2, 0.4),
            make_row(-0.1, -0.2, 0.4, 0.3, 0.5),
        ]
    )
    summary = build_summary(rows)
    logic = summary["per_axis"]["logic"]
    assert abs(logic["stable_bridge"]["share_within_parent"] - 0.6) < 1e-9
    assert abs(logic["fragile_bridge"]["share_within_parent"] - 0.4) < 1e-9
    assert logic["constraint_conflict"]["nonzero_count"] == 1
    assert logic["destructive_conflict"]["nonzero_count"] == 1
