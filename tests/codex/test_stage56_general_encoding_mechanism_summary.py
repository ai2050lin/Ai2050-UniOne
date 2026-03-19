from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_general_encoding_mechanism_summary import build_summary  # noqa: E402


def test_build_summary_marks_syntax_prompt_contaminated_when_prompt_share_is_higher() -> None:
    summary = build_summary(
        structure_atlas={
            "system_claim": "system",
            "parts_of_speech": {
                "nouns": {"claim": "noun"},
                "adjectives": {"claim": "adj"},
                "verbs": {"claim": "verb"},
                "abstract_nouns": {"claim": "abs"},
                "adverbs": {"claim": "adv"},
            },
        },
        relation_summary={
            "group_count": 10,
            "counts_by_interpretation": {"local_linear": 2, "path_bundle": 7, "control_mixed": 1},
        },
        wordclass_summary={"probes": {"adjective": {"primary_mechanism": "modifier_fiber"}}},
        unified_atlas={
            "common_laws": {
                "shared_closure_categories": ["fruit"],
                "control_laws": {"logic_prototype_to_synergy_corr": 0.2},
            }
        },
        window_summary={
            "per_component": {
                "logic_prototype": {
                    "hidden_to_joint_adv": {"best_window": "tail_pos_-9..tail_pos_-8", "best_corr": 0.5},
                    "mlp_to_joint_adv": {"best_window": "tail_pos_-9..tail_pos_-8", "best_corr": 0.5},
                },
                "logic_fragile_bridge": {
                    "hidden_to_synergy": {"best_window": "tail_pos_-2..tail_pos_-1", "best_corr": -0.2},
                    "hidden_to_joint_adv": {"best_window": "tail_pos_-2..tail_pos_-1", "best_corr": -0.4},
                },
                "syntax_constraint_conflict": {
                    "hidden_to_synergy": {"best_window": "tail_pos_-8..tail_pos_-5", "best_corr": 0.8},
                    "mlp_to_synergy": {"best_window": "tail_pos_-6..tail_pos_-3", "best_corr": 0.7},
                },
            }
        },
        natural_decoupling={
            "per_axis": {
                "style": {"mean_hidden_generated_share": 0.7},
                "logic": {"mean_hidden_generated_share": 0.6},
                "syntax": {"mean_hidden_generated_share": 0.2, "mean_hidden_prompt_share": 0.8},
            },
            "per_component": {
                "syntax_constraint_conflict": {
                    "weighted_hidden_prompt_share": 0.9,
                    "weighted_hidden_generated_share": 0.1,
                }
            },
        },
    )
    assert summary["natural_generation_decoupling"]["judgement"] == "syntax_prompt_contaminated"
    assert summary["system_encoding_law"]["dominant_form"] == "anchor_fiber_path_bundle_with_windowed_closure"
