from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_multimodel_language_unified_atlas import build_common_laws, build_model_private  # noqa: E402


def test_build_common_laws_extracts_control_signals() -> None:
    laws = build_common_laws(
        relation_summary={"counts_by_interpretation": {"local_linear": 10, "path_bundle": 20}},
        probe_summary={"probes": {"verb": {"primary_mechanism": "transport"}}},
        discovery_summary={"top_consensus_categories": [{"category": "fruit"}, {"category": "action"}]},
        pair_link_summary={"axis_target_stats": {"logic": {"prototype_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": 0.4}}}}, "syntax": {"conflict_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": 0.27}}}}}},
        bxm_summary={"per_axis": {"logic": {"fragile_bridge": {"corr_to_union_synergy_joint": -0.4}}}},
    )
    assert laws["control_laws"]["logic_prototype_to_synergy_corr"] == 0.4
    assert laws["relation_system_split"]["path_bundle_count"] == 20


def test_build_model_private_assigns_readings() -> None:
    result = build_model_private(
        [
            {"model_id": "Qwen/Qwen3-4B", "strict_positive_pair_ratio": 0.4, "mean_union_joint_adv": 0.1, "mean_union_synergy_joint": -0.02, "top_prototype_layers": [{"layer": 35}], "strict_positive_categories": ["action"]},
            {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "strict_positive_pair_ratio": 0.08, "mean_union_joint_adv": -0.03, "mean_union_synergy_joint": -0.06, "top_prototype_layers": [{"layer": 27}], "strict_positive_categories": ["weather"]},
        ],
        {"per_model": {}},
    )
    assert result["Qwen/Qwen3-4B"]["reading"] == "闭包友好型实现"
    assert result["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]["reading"] == "结构稳定但协同偏负型实现"
