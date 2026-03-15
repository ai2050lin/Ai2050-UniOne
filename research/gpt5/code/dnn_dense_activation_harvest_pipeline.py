from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from research.gpt5.code.dnn_direct_dense_harvest_manifest import DirectDenseHarvestManifest


@dataclass
class HarvestSourceScript:
    relative_path: str
    model_scope: str
    capture_mode: str
    target_view: str
    supports_direct_dense: bool
    exactness_tier: str
    notes: str

    def to_dict(self, root: Path) -> Dict[str, object]:
        path = root / self.relative_path
        return {
            "relative_path": self.relative_path,
            "exists": path.exists(),
            "model_scope": self.model_scope,
            "capture_mode": self.capture_mode,
            "target_view": self.target_view,
            "supports_direct_dense": self.supports_direct_dense,
            "exactness_tier": self.exactness_tier,
            "notes": self.notes,
        }


@dataclass
class HarvestTask:
    bucket_name: str
    priority: str
    target_units: int
    target_kind: str
    concept_groups: List[str]
    prompt_families: List[str]
    capture_sites: List[str]
    tensor_layout: str
    scripts: List[HarvestSourceScript]
    blocking_gaps: List[str]

    def summary(self, root: Path) -> Dict[str, object]:
        script_rows = [script.to_dict(root) for script in self.scripts]
        existing_paths = sum(1 for row in script_rows if row["exists"])
        direct_dense_paths = sum(
            1 for row in script_rows if row["exists"] and row["supports_direct_dense"]
        )
        extraction_path_exists = existing_paths == len(script_rows) and len(script_rows) > 0
        launch_ready = extraction_path_exists and direct_dense_paths >= 1 and len(self.concept_groups) >= 2
        readiness = min(
            1.0,
            0.30 * min(1.0, existing_paths / max(1, len(script_rows)))
            + 0.25 * min(1.0, direct_dense_paths / max(1, len(script_rows)))
            + 0.20 * min(1.0, len(self.concept_groups) / 3.0)
            + 0.15 * min(1.0, len(self.capture_sites) / 3.0)
            + 0.10 * min(1.0, self.target_units / 200.0),
        )
        return {
            "bucket_name": self.bucket_name,
            "priority": self.priority,
            "target_units": self.target_units,
            "target_kind": self.target_kind,
            "concept_groups": self.concept_groups,
            "prompt_families": self.prompt_families,
            "capture_sites": self.capture_sites,
            "tensor_layout": self.tensor_layout,
            "script_count": len(script_rows),
            "existing_script_count": existing_paths,
            "direct_dense_script_count": direct_dense_paths,
            "extraction_path_exists": extraction_path_exists,
            "launch_ready": launch_ready,
            "task_readiness": float(readiness),
            "blocking_gaps": self.blocking_gaps,
            "scripts": script_rows,
        }


@dataclass
class DenseActivationHarvestPipeline:
    tasks: Dict[str, HarvestTask]

    @classmethod
    def from_repo(cls, root: Path) -> "DenseActivationHarvestPipeline":
        manifest = DirectDenseHarvestManifest.from_artifacts(root)
        bucket_map = manifest.buckets

        tasks = {
            "specific_dense_signature": HarvestTask(
                bucket_name="specific_dense_signature",
                priority=bucket_map["specific_dense_signature"].priority,
                target_units=bucket_map["specific_dense_signature"].target_units,
                target_kind=bucket_map["specific_dense_signature"].target_kind,
                concept_groups=["fruit_specific", "animal_specific", "micro_attributes"],
                prompt_families=["identity_prompts", "attribute_prompts", "contrast_prompts", "recall_prompts"],
                capture_sites=["mlp.gate_proj", "last_token_gate_vector", "causal_ablation_subset"],
                tensor_layout="[num_prompts, num_layers, d_ff] and flattened [num_prompts, num_layers*d_ff]",
                scripts=[
                    HarvestSourceScript(
                        relative_path="tests/codex/deepseek7b_apple_neuron_ablation.py",
                        model_scope="deepseek_7b",
                        capture_mode="forward_hook_mlp_gate",
                        target_view="apple concept-specific gate activations",
                        supports_direct_dense=True,
                        exactness_tier="direct_dense",
                        notes="已有 GateCollector 与单神经元消融，可直接复用为 dense specific 采样入口。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/deepseek7b_apple_triscale_micro_causal.py",
                        model_scope="deepseek_7b",
                        capture_mode="forward_hook_mlp_gate",
                        target_view="micro/meso/macro triscale candidate neurons",
                        supports_direct_dense=True,
                        exactness_tier="direct_dense",
                        notes="可把 micro 属性维与 meso 概念维统一到同一 dense signature 格式。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/deepseek7b_cat_dog_attribute_causal.py",
                        model_scope="deepseek_7b",
                        capture_mode="forward_hook_mlp_gate",
                        target_view="animal attribute-specific dense candidates",
                        supports_direct_dense=True,
                        exactness_tier="direct_dense",
                        notes="补足非水果家族的 concept-specific 与 attribute-specific 真实路径。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/deepseek7b_apple_100_concepts_compare.py",
                        model_scope="deepseek_7b",
                        capture_mode="forward_hook_mlp_gate",
                        target_view="large concept comparison matrix",
                        supports_direct_dense=True,
                        exactness_tier="direct_dense",
                        notes="可扩成大样本 concept-specific dense harvesting 队列。",
                    ),
                ],
                blocking_gaps=[
                    "当前仍以 DeepSeek 路线为主，Qwen 的 direct dense specific 采集脚手架不足。",
                    "还没有把多个 concept-specific hook 输出统一落成标准 dense tensor 存储格式。",
                ],
            ),
            "protocol_dense_signature": HarvestTask(
                bucket_name="protocol_dense_signature",
                priority=bucket_map["protocol_dense_signature"].priority,
                target_units=bucket_map["protocol_dense_signature"].target_units,
                target_kind=bucket_map["protocol_dense_signature"].target_kind,
                concept_groups=["fruit_protocol", "animal_protocol", "abstract_protocol"],
                prompt_families=["field_prompts", "boundary_prompts", "tool_interface_prompts"],
                capture_sites=["attention_heads", "layer_head_usage", "protocol_boundary_scan"],
                tensor_layout="[num_prompts, num_layers, num_heads] with per-field margins",
                scripts=[
                    HarvestSourceScript(
                        relative_path="tests/codex/test_qwen3_deepseek7b_concept_protocol_field_mapping.py",
                        model_scope="qwen3_4b+deepseek_7b",
                        capture_mode="attention_head_usage",
                        target_view="concept-to-protocol-field head map",
                        supports_direct_dense=True,
                        exactness_tier="head_dense",
                        notes="已经能给出 U(c,tau,l,h) 级别的协议字段使用强度。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/test_qwen3_deepseek7b_protocol_field_boundary_atlas.py",
                        model_scope="qwen3_4b+deepseek_7b",
                        capture_mode="protocol_boundary_scan",
                        target_view="boundary causal margins by field",
                        supports_direct_dense=True,
                        exactness_tier="head_dense",
                        notes="可直接形成 protocol boundary dense coordinates。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/test_qwen3_deepseek7b_hard_online_tool_interface.py",
                        model_scope="qwen3_4b+deepseek_7b",
                        capture_mode="protocol_interface_probe",
                        target_view="tool-interface protocol activation",
                        supports_direct_dense=False,
                        exactness_tier="summary_proxy",
                        notes="当前更偏行为接口验证，仍需补成统一 exact tensor 导出。",
                    ),
                ],
                blocking_gaps=[
                    "协议字段已有 head-level 路径，但还没有统一导出为 neuron-equivalent dense signature corpus。",
                    "tool/interface 路线仍偏 proxy 验证，缺少与 field head map 对齐的 exact tensor 落盘。",
                ],
            ),
            "topology_dense_signature": HarvestTask(
                bucket_name="topology_dense_signature",
                priority=bucket_map["topology_dense_signature"].priority,
                target_units=bucket_map["topology_dense_signature"].target_units,
                target_kind=bucket_map["topology_dense_signature"].target_kind,
                concept_groups=["family_topology", "relation_topology", "attention_topology"],
                prompt_families=["family_membership_prompts", "relation_prompts", "wrong_family_controls"],
                capture_sites=["attention_topology", "relation_topology", "family_margin_scan"],
                tensor_layout="[num_prompts, num_layers, num_heads] plus family margin tables",
                scripts=[
                    HarvestSourceScript(
                        relative_path="tests/codex/test_qwen3_deepseek7b_attention_topology_atlas.py",
                        model_scope="qwen3_4b+deepseek_7b",
                        capture_mode="attention_topology_scan",
                        target_view="family/topology atlas",
                        supports_direct_dense=True,
                        exactness_tier="head_dense",
                        notes="注意力拓扑已经能给出 family 对错分界 margin。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/test_qwen3_deepseek7b_relation_topology_atlas.py",
                        model_scope="qwen3_4b+deepseek_7b",
                        capture_mode="relation_topology_scan",
                        target_view="relation topology atlas",
                        supports_direct_dense=True,
                        exactness_tier="head_dense",
                        notes="关系拓扑可补 relation-context fiber 的 dense topology 入口。",
                    ),
                ],
                blocking_gaps=[
                    "当前 topology 仍以 head-level rows 为主，缺少更底层 neuron-equivalent dense projection。",
                ],
            ),
            "successor_dense_signature": HarvestTask(
                bucket_name="successor_dense_signature",
                priority=bucket_map["successor_dense_signature"].priority,
                target_units=bucket_map["successor_dense_signature"].target_units,
                target_kind=bucket_map["successor_dense_signature"].target_kind,
                concept_groups=["temporal_chain_reasoning", "online_recovery_chain", "successor_stage_paths"],
                prompt_families=["multi_hop_chain_prompts", "temporal_stage_prompts", "protocol_successor_prompts"],
                capture_sites=["mlp.gate_proj", "temporal_stage_inventory", "online_recovery_chain"],
                tensor_layout="[num_chains, num_stages, num_layers, d_ff] with chain-level margins",
                scripts=[
                    HarvestSourceScript(
                        relative_path="tests/codex/deepseek7b_multihop_reasoning_route_test.py",
                        model_scope="deepseek_7b",
                        capture_mode="forward_hook_mlp_gate",
                        target_view="multi-hop route neurons",
                        supports_direct_dense=True,
                        exactness_tier="direct_dense",
                        notes="已有链路类 gate 激活与最小因果子集，是 dense successor 的直接入口。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/test_qwen3_deepseek7b_online_recovery_chain.py",
                        model_scope="qwen3_4b+deepseek_7b",
                        capture_mode="online_recovery_probe",
                        target_view="recovery-chain stage metrics",
                        supports_direct_dense=False,
                        exactness_tier="summary_proxy",
                        notes="已经覆盖在线恢复链，但当前主要输出 summary 指标，还需补 dense activations。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/test_theory_track_successor_strengthened_reasoning_inventory.py",
                        model_scope="theory_track",
                        capture_mode="successor_inventory_builder",
                        target_view="chain/stage inventory",
                        supports_direct_dense=False,
                        exactness_tier="inventory_proxy",
                        notes="给出 successor 的链条和阶段范围，是 dense chain harvesting 的任务编排底座。",
                    ),
                ],
                blocking_gaps=[
                    "successor 目前仍主要停在链条 inventory 和 route summary，缺少统一 dense chain tensor 导出。",
                    "Qwen 侧 successor 直采路径弱于 DeepSeek，跨模型对齐还不够。",
                ],
            ),
            "lift_dense_signature": HarvestTask(
                bucket_name="lift_dense_signature",
                priority=bucket_map["lift_dense_signature"].priority,
                target_units=bucket_map["lift_dense_signature"].target_units,
                target_kind=bucket_map["lift_dense_signature"].target_kind,
                concept_groups=["meso_to_macro_lift", "abstraction_ladder", "role_bridge"],
                prompt_families=["category_bridge_prompts", "abstraction_prompts", "role_bridge_prompts"],
                capture_sites=["category_bridge", "abstraction_ladder", "macro_role_mapping"],
                tensor_layout="[num_levels, num_prompts, num_layers, hidden_or_gate_dim]",
                scripts=[
                    HarvestSourceScript(
                        relative_path="tests/codex/test_qwen_deepseek_micro_meso_macro_encoding_map.py",
                        model_scope="qwen3_4b+deepseek_7b",
                        capture_mode="micro_meso_macro_map",
                        target_view="meso-to-macro lift map",
                        supports_direct_dense=False,
                        exactness_tier="summary_proxy",
                        notes="当前能提供 lift 的系统对象定义，但还缺 dense neuron 级导出。",
                    ),
                    HarvestSourceScript(
                        relative_path="tests/codex/test_gpt2_qwen3_basis_hierarchy_compare.py",
                        model_scope="gpt2+qwen3_4b",
                        capture_mode="basis_hierarchy_compare",
                        target_view="hierarchy lift comparison",
                        supports_direct_dense=False,
                        exactness_tier="summary_proxy",
                        notes="有层级比较，但尚未收束成 lift dense signature pipeline。",
                    ),
                ],
                blocking_gaps=[
                    "lift 目前更像结构映射和层级比较，缺少直接的 dense activation harvesting 入口。",
                ],
            ),
        }
        return cls(tasks=tasks)

    def summary(self, root: Path) -> Dict[str, object]:
        task_rows = {name: task.summary(root) for name, task in self.tasks.items()}
        all_scripts = sum(row["script_count"] for row in task_rows.values())
        existing_scripts = sum(row["existing_script_count"] for row in task_rows.values())
        highest_priority_rows = [row for row in task_rows.values() if row["priority"] == "highest"]
        runnable_highest_priority = sum(1 for row in highest_priority_rows if row["launch_ready"])
        ready_tasks = sum(1 for row in task_rows.values() if row["task_readiness"] >= 0.80)
        direct_dense_coverage = sum(row["direct_dense_script_count"] for row in task_rows.values())
        pipeline_ready_score = min(
            1.0,
            0.25 * min(1.0, existing_scripts / max(1, all_scripts))
            + 0.25 * min(1.0, ready_tasks / max(1, len(task_rows)))
            + 0.25 * min(1.0, runnable_highest_priority / 3.0)
            + 0.15 * min(1.0, direct_dense_coverage / 8.0)
            + 0.10,
        )
        return {
            "total_buckets": len(task_rows),
            "highest_priority_bucket_count": len(highest_priority_rows),
            "ready_bucket_count": ready_tasks,
            "runnable_highest_priority_bucket_count": runnable_highest_priority,
            "total_source_scripts": all_scripts,
            "existing_source_scripts": existing_scripts,
            "direct_dense_script_coverage": direct_dense_coverage,
            "pipeline_ready_score": float(pipeline_ready_score),
            "tasks": task_rows,
        }
