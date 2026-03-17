from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage5_readout_coupled_search import (  # noqa: E402
    build_prototype_proxy_rows,
    candidate_allowed_in_lane,
    choose_representative_baselines,
    effect_score,
    is_category_word,
    is_prototype_proxy_row,
    lane_matches,
    limit_candidate_indices,
    normalize_lexeme,
    selection_score,
    select_stage4_candidates,
    strict_effect_score,
    subset_overlap_ratio,
)


def test_select_stage4_candidates_prefers_joint_hits_and_respects_category_limit():
    rows = [
        {
            "item": {"term": "a", "category": "tech"},
            "source_kind": "combined",
            "subset_size": 32,
            "pair_metrics": {"joint_binding_hit": True, "joint_adv_score": 0.5, "margin_adv_vs_random": 0.2, "category_adv_vs_random": 0.001},
        },
        {
            "item": {"term": "b", "category": "tech"},
            "source_kind": "combined",
            "subset_size": 32,
            "pair_metrics": {"joint_binding_hit": True, "joint_adv_score": 0.4, "margin_adv_vs_random": 0.1, "category_adv_vs_random": 0.001},
        },
        {
            "item": {"term": "c", "category": "animal"},
            "source_kind": "family_shared",
            "subset_size": 48,
            "pair_metrics": {"joint_binding_hit": False, "joint_adv_score": 0.6, "margin_adv_vs_random": 0.3, "category_adv_vs_random": 0.0},
        },
    ]
    picked = select_stage4_candidates(rows, max_candidates=3, per_category_limit=1)
    assert [row["item"]["term"] for row in picked] == ["a", "c"]


def test_select_stage4_candidates_can_force_category_coverage():
    rows = [
        {
            "item": {"term": "a", "category": "tech"},
            "source_kind": "combined",
            "subset_size": 32,
            "subset_flat_indices": [1, 2, 3],
            "pair_metrics": {"joint_binding_hit": True, "joint_adv_score": 0.9, "margin_adv_vs_random": 0.3, "category_adv_vs_random": 0.001},
        },
        {
            "item": {"term": "b", "category": "tech"},
            "source_kind": "family_shared",
            "subset_size": 32,
            "subset_flat_indices": [4, 5, 6],
            "pair_metrics": {"joint_binding_hit": True, "joint_adv_score": 0.8, "margin_adv_vs_random": 0.2, "category_adv_vs_random": 0.001},
        },
        {
            "item": {"term": "c", "category": "human"},
            "source_kind": "combined",
            "subset_size": 24,
            "subset_flat_indices": [10, 11, 12],
            "pair_metrics": {"joint_binding_hit": False, "joint_adv_score": 0.05, "margin_adv_vs_random": 0.01, "category_adv_vs_random": 0.0},
        },
    ]
    picked = select_stage4_candidates(
        rows,
        max_candidates=2,
        per_category_limit=2,
        require_category_coverage=True,
    )
    assert [row["item"]["term"] for row in picked] == ["a", "c"]


def test_select_stage4_candidates_avoids_heavy_overlap_when_possible():
    rows = [
        {
            "item": {"term": "a", "category": "tech"},
            "source_kind": "combined",
            "subset_size": 32,
            "subset_flat_indices": [1, 2, 3, 4],
            "pair_metrics": {"joint_binding_hit": True, "joint_adv_score": 0.40, "margin_adv_vs_random": 0.2, "category_adv_vs_random": 0.001},
        },
        {
            "item": {"term": "b", "category": "animal"},
            "source_kind": "combined",
            "subset_size": 32,
            "subset_flat_indices": [1, 2, 3, 5],
            "pair_metrics": {"joint_binding_hit": True, "joint_adv_score": 0.39, "margin_adv_vs_random": 0.2, "category_adv_vs_random": 0.001},
        },
        {
            "item": {"term": "c", "category": "animal"},
            "source_kind": "family_shared",
            "subset_size": 32,
            "subset_flat_indices": [10, 11, 12, 13],
            "pair_metrics": {"joint_binding_hit": False, "joint_adv_score": 0.20, "margin_adv_vs_random": 0.1, "category_adv_vs_random": 0.0},
        },
    ]
    picked = select_stage4_candidates(rows, max_candidates=2, per_category_limit=1, max_overlap=0.5)
    assert [row["item"]["term"] for row in picked] == ["a", "c"]


def test_limit_candidate_indices_prefers_high_baseline_importance():
    row = {"subset_flat_indices": [8, 5, 7, 9]}
    baseline_signature = {
        "signature_top_indices": [7, 9, 100],
        "signature_top_values": [0.9, 0.8, 0.1],
    }
    assert limit_candidate_indices(row, baseline_signature, max_neurons=3) == [7, 9, 8]


def test_limit_candidate_indices_respects_layer_cap_before_filling():
    row = {"subset_flat_indices": [27, 26, 25, 24, 3, 2]}
    baseline_signature = {
        "signature_top_indices": [27, 26, 25, 24, 3, 2],
        "signature_top_values": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    }
    assert limit_candidate_indices(row, baseline_signature, max_neurons=4, d_ff=10, max_per_layer=2) == [27, 26, 3, 2]


def test_subset_overlap_ratio_uses_jaccard():
    left = {"subset_flat_indices": [1, 2, 3]}
    right = {"subset_flat_indices": [2, 3, 4, 5]}
    assert subset_overlap_ratio(left, right) == 0.4


def test_effect_score_uses_margin_plus_alpha_times_category():
    assert effect_score(0.01, 0.001, 256.0) == 0.266


def test_normalize_lexeme_and_category_word_detection():
    assert normalize_lexeme("Animal!") == "animal"
    assert is_category_word({"term": "animal", "category": "animal"}) is True
    assert is_category_word({"term": "animals", "category": "animal"}) is True
    assert is_category_word({"term": "kangaroo", "category": "animal"}) is False


def test_selection_score_penalizes_category_words():
    row = {
        "item": {"term": "animal", "category": "animal"},
        "pair_metrics": {"joint_adv_score": 0.5, "margin_adv_vs_random": 0.2, "joint_binding_hit": True},
    }
    plain = selection_score(row, [], overlap_penalty=0.15, category_word_penalty=0.0)
    penalized = selection_score(row, [], overlap_penalty=0.15, category_word_penalty=0.3)
    assert penalized == plain - 0.3


def test_selection_score_penalizes_zero_margin_adv_rows():
    row = {
        "item": {"term": "animal", "category": "animal"},
        "pair_metrics": {
            "joint_adv_score": 0.5,
            "margin_adv_vs_random": 0.0,
            "joint_binding_hit": True,
        },
    }
    plain = selection_score(row, [], overlap_penalty=0.15, margin_adv_penalty=0.0)
    penalized = selection_score(
        row,
        [],
        overlap_penalty=0.15,
        margin_adv_threshold=0.0,
        margin_adv_penalty=0.05,
    )
    assert penalized == plain - 0.05


def test_strict_effect_score_penalizes_weak_margin_adv():
    plain = strict_effect_score(0.0, 0.001, 256.0, margin_adv_penalty=0.0)
    penalized = strict_effect_score(0.0, 0.001, 256.0, margin_adv_penalty=0.05)
    assert penalized == plain - 0.05


def test_lane_matches_splits_prototype_and_instance():
    proto = {"term": "animal", "category": "animal"}
    inst = {"term": "kangaroo", "category": "animal"}
    assert lane_matches(proto, "prototype") is True
    assert lane_matches(proto, "instance") is False
    assert lane_matches(inst, "prototype") is False
    assert lane_matches(inst, "instance") is True
    assert lane_matches(inst, "mixed") is True


def test_lane_matches_accepts_family_prototype_proxy():
    inst = {"term": "kangaroo", "category": "animal"}
    assert lane_matches(inst, "prototype", source_kind="family_prototype") is True
    assert lane_matches(inst, "instance", source_kind="family_prototype") is False


def test_candidate_allowed_in_lane_can_forbid_prototype_proxy():
    inst = {"term": "kangaroo", "category": "animal"}
    assert candidate_allowed_in_lane(
        inst,
        "prototype",
        source_kind="family_prototype",
        allow_prototype_proxy=False,
    ) is False


def test_candidate_allowed_in_lane_can_require_real_category_words():
    proto = {"term": "animal", "category": "animal"}
    inst = {"term": "kangaroo", "category": "animal"}
    assert candidate_allowed_in_lane(
        proto,
        "prototype",
        source_kind="combined",
        prototype_term_mode="category_only",
    ) is True
    assert candidate_allowed_in_lane(
        inst,
        "prototype",
        source_kind="combined",
        prototype_term_mode="category_only",
    ) is False


def test_choose_representative_baselines_prefers_stronger_category_margin():
    baselines = [
        {
            "item": {"term": "clerk", "category": "human"},
            "baseline_readout": {"category_margin": 0.01, "correct_prob": 0.10},
        },
        {
            "item": {"term": "programmer", "category": "human"},
            "baseline_readout": {"category_margin": 0.03, "correct_prob": 0.05},
        },
    ]
    chosen = choose_representative_baselines(baselines, ["human"])
    assert chosen["human"]["item"]["term"] == "programmer"


def test_build_prototype_proxy_rows_uses_closure_family_and_marks_proxy():
    families = [
        {
            "record_type": "family_prototype",
            "pool": "closure",
            "category": "human",
            "prototype_top_indices": [9, 8, 7, 6],
            "mean_prompt_stability": 0.72,
        }
    ]
    baselines = [
        {
            "item": {"term": "programmer", "category": "human", "role": "anchor"},
            "baseline_readout": {"category_margin": 0.03, "correct_prob": 0.05},
        }
    ]
    rows = build_prototype_proxy_rows(families, baselines, ["human"])
    assert len(rows) == 1
    row = rows[0]
    assert row["item"]["term"] == "programmer"
    assert row["source_kind"] == "family_prototype"
    assert row["subset_flat_indices"] == [9, 8, 7, 6]
    assert is_prototype_proxy_row(row) is True
