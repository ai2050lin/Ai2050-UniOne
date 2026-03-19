from __future__ import annotations

from stage56_frontier_heterogeneity_split import build_rows, build_summary


def test_build_rows_computes_frontier_terms() -> None:
    rows = build_rows(
        [
            {
                "model_id": "m",
                "category": "c",
                "prototype_term": "p",
                "instance_term": "i",
                "union_joint_adv": 1.0,
                "union_synergy_joint": 2.0,
                "strict_positive_synergy": True,
                "axes": {
                    "style": {
                        "pair_compaction_middle_mean": 1.0,
                        "pair_coverage_middle_mean": 3.0,
                        "role_asymmetry_compaction_l1": 0.1,
                        "role_asymmetry_coverage_l1": 0.2,
                        "pair_compaction_early_mean": 0.5,
                        "pair_compaction_late_mean": 1.5,
                        "pair_coverage_early_mean": 2.0,
                        "pair_coverage_late_mean": 3.0,
                    },
                    "logic": {
                        "pair_compaction_middle_mean": 1.0,
                        "pair_coverage_middle_mean": 3.0,
                        "role_asymmetry_compaction_l1": 0.1,
                        "role_asymmetry_coverage_l1": 0.2,
                        "pair_compaction_early_mean": 0.5,
                        "pair_compaction_late_mean": 1.5,
                        "pair_coverage_early_mean": 2.0,
                        "pair_coverage_late_mean": 3.0,
                    },
                    "syntax": {
                        "pair_compaction_middle_mean": 1.0,
                        "pair_coverage_middle_mean": 3.0,
                        "role_asymmetry_compaction_l1": 0.1,
                        "role_asymmetry_coverage_l1": 0.2,
                        "pair_compaction_early_mean": 0.5,
                        "pair_compaction_late_mean": 1.5,
                        "pair_coverage_early_mean": 2.0,
                        "pair_coverage_late_mean": 3.0,
                    },
                },
            }
        ]
    )
    row = rows[0]
    assert row["frontier_compaction_term"] == 1.0
    assert row["frontier_coverage_term"] == 3.0
    assert row["frontier_balance_term"] == 2.0


def test_build_summary_keeps_stable_features() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        rows.append(
            {
                "frontier_compaction_term": -scale,
                "frontier_coverage_term": scale,
                "frontier_separation_term": -scale,
                "frontier_compaction_late_shift": scale,
                "frontier_coverage_late_shift": scale,
                "frontier_balance_term": scale,
                "union_joint_adv": scale,
                "union_synergy_joint": scale,
                "strict_positive_synergy": scale,
            }
        )
    summary = build_summary(rows)
    features = {row["feature"]: row["sign"] for row in summary["stable_features"]}
    assert features["frontier_coverage_term"] == "positive"
