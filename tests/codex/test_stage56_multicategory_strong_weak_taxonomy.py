from __future__ import annotations

from stage56_multicategory_strong_weak_taxonomy import (
    build_case_groups,
    dominant_structure,
    parse_case_group_spec,
)


def test_dominant_structure_bridge_case() -> None:
    best_strong = {"metrics": {"joint_adv_mean": 0.01}}
    best_weak = {"metrics": {"joint_adv_mean": -0.01}}
    best_mixed = {"metrics": {"joint_adv_mean": 0.03}}
    assert dominant_structure("weak_bridge_positive", best_strong, best_weak, best_mixed) == "bridge_dominant"


def test_dominant_structure_strong_core_case() -> None:
    best_strong = {"metrics": {"joint_adv_mean": 0.04}}
    best_weak = {"metrics": {"joint_adv_mean": 0.01}}
    best_mixed = {"metrics": {"joint_adv_mean": 0.02}}
    assert dominant_structure("weak_drag_or_conflict", best_strong, best_weak, best_mixed) == "strong_core_dominant"


def test_parse_case_group_spec_reads_label_model_and_root() -> None:
    row = parse_case_group_spec("glm_real|zai-org/GLM-4-9B-Chat-HF|D:/tmp/glm")
    assert row["label"] == "glm_real"
    assert row["model_id"] == "zai-org/GLM-4-9B-Chat-HF"
    assert str(row["model_root"]).endswith("D:\\tmp\\glm")


def test_build_case_groups_keeps_existing_labels_unique() -> None:
    groups = build_case_groups([
        "glm_real|zai-org/GLM-4-9B-Chat-HF|D:/tmp/glm",
        "custom_real|Qwen/Qwen3-4B|D:/tmp/qwen",
    ])
    labels = [str(row["label"]) for row in groups]
    assert labels.count("glm_real") <= 1
    assert "custom_real" in labels


def test_build_case_groups_can_skip_default_groups() -> None:
    groups = build_case_groups(
        ["seq_qwen|Qwen/Qwen3-4B|D:/tmp/qwen_seq"],
        include_defaults=False,
    )
    assert [str(row["label"]) for row in groups] == ["seq_qwen"]
