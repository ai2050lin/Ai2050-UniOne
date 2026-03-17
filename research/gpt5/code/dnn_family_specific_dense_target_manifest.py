from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class DnnFamilySpecificDenseTargetManifest:
    sprint_manifest: Dict[str, Any]
    schema_block: Dict[str, Any]
    specific_bridge: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnFamilySpecificDenseTargetManifest":
        temp = root / "tests" / "codex_temp"
        return cls(
            sprint_manifest=load_json(temp / "dnn_joint_closure_sprint_manifest_block_20260315.json"),
            schema_block=load_json(temp / "dnn_joint_dense_export_schema_block_20260315.json"),
            specific_bridge=load_json(temp / "dnn_specific_math_bridge_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        sprint_axes = self.sprint_manifest["headline_metrics"]["sprint_axes"]
        specific_axis = next(axis for axis in sprint_axes if axis["recommended_bucket"] == "specific_dense_signature")
        schema_rows = self.schema_block["headline_metrics"]["schema_rows"]
        specific_schema = next(row for row in schema_rows if row["bucket_name"] == "specific_dense_signature")
        bridge = self.specific_bridge["headline_metrics"]

        closure_gap = float(1.0 - bridge["exact_specific_closure_score"])
        readiness = float(specific_axis["task_readiness"])
        projected_dense_uplift = min(1.0, 0.55 * readiness + 0.45 * closure_gap)

        target_groups: List[Dict[str, Any]] = [
            {
                "name": "fruit_family_specific_core",
                "priority_rank": 1,
                "concepts": ["apple", "banana", "pear", "orange", "grape"],
                "why": "先在同一家族里把 family patch 和 concept offset 的主轴打穿，这是 family-to-specific exact closure 的最短路径。",
                "prompt_families": ["identity_prompts", "contrast_prompts", "attribute_prompts"],
                "capture_axes": ["prompt_id", "layer", "neuron", "concept", "category", "activation_tensor"],
                "expected_gain": "直接提高 family -> specific 的可恢复性。",
            },
            {
                "name": "cross_family_shape_controls",
                "priority_rank": 2,
                "concepts": ["apple", "moon", "orange", "sun"],
                "why": "用跨家族但带相似属性的概念做对照，区分“共享属性方向”与“对象身份底板”。",
                "prompt_families": ["attribute_prompts", "contrast_prompts", "wrong_family_controls"],
                "capture_axes": ["prompt_id", "layer", "neuron", "concept", "category", "activation_tensor"],
                "expected_gain": "帮助分离 attribute fiber 和 family patch，避免把相似属性误判成相同对象编码。",
            },
            {
                "name": "family_neighbor_recovery",
                "priority_rank": 3,
                "concepts": ["apple", "pear", "banana"],
                "why": "直接检查苹果是否能从 fruit 底板恢复到具体邻近对象差异，这是最贴近当前 exact gap 的局部闭环测试。",
                "prompt_families": ["recall_prompts", "identity_prompts", "contrast_prompts"],
                "capture_axes": ["prompt_id", "layer", "neuron", "concept", "category", "activation_tensor"],
                "expected_gain": "细化 apple -> pear / banana 的局部恢复边界。",
            },
        ]

        metric_lines_cn = [
            f"（family到specific闭合缺口）family_to_specific_gap = {bridge['family_to_specific_gap']:.4f}",
            f"（specific精确闭合度）exact_specific_closure_score = {bridge['exact_specific_closure_score']:.4f}",
            f"（specific任务准备度）specific_task_readiness = {readiness:.4f}",
            f"（specific导出schema可启动）specific_schema_launchable = {1.0 if specific_schema['launchable'] else 0.0:.4f}",
            f"（本轮dense目标预估提升）projected_dense_uplift = {projected_dense_uplift:.4f}",
            f"（dense目标组数量）dense_target_group_count = {len(target_groups):.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "target_groups": target_groups,
            "specific_schema": specific_schema,
            "closure_gap": closure_gap,
            "projected_dense_uplift": projected_dense_uplift,
            "strict_conclusion": {
                "core_answer": "The next concrete step for family-to-specific exact closure is no longer abstract: first harvest dense tensors on fruit-family specific targets, then add cross-family shape controls, and only then judge whether shared attributes and object-specific offsets can be cleanly separated.",
                "main_hard_gaps": [
                    "family-to-specific exact closure is still far below parameteric restoration",
                    "cross-family attribute controls are still missing direct dense tensors",
                    "current evidence is still stronger at row/signature level than dense neuron level",
                ],
            },
        }
