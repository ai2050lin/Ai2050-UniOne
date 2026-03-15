from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class HarvestBucket:
    name: str
    target_units: int
    target_kind: str
    priority: str
    rationale: str


@dataclass
class DirectDenseHarvestManifest:
    buckets: Dict[str, HarvestBucket]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DirectDenseHarvestManifest":
        temp = root / "tests" / "codex_temp"
        codebook = load_json(temp / "concept_family_unified_codebook_20260308.json")
        protocol_map = load_json(temp / "qwen3_deepseek7b_concept_protocol_field_mapping_20260309.json")
        protocol_boundary = load_json(temp / "qwen3_deepseek7b_protocol_field_boundary_atlas_20260309.json")
        attention_topology = load_json(temp / "qwen3_deepseek7b_attention_topology_atlas_20260309.json")
        relation_topology = load_json(temp / "qwen3_deepseek7b_relation_topology_atlas_20260309.json")
        successor_inventory = load_json(temp / "theory_track_successor_strengthened_reasoning_inventory_20260312.json")
        category_bridge = load_json(temp / "category_abstraction_bridge_20260308.json")
        abstraction_ladder = load_json(temp / "abstraction_ladder_hierarchy_20260308.json")

        specific_units = sum(len(row["concept_specific"]) for row in codebook["family_stats"].values())
        specific_units += sum(len(model["concepts"]) for model in protocol_map["models"].values())
        specific_units += sum(len(model["concepts"]) for model in attention_topology["models"].values())

        protocol_units = sum(len(model["concepts"]) for model in protocol_map["models"].values())
        protocol_units += sum(len(model["concepts"]) for model in protocol_boundary["models"].values())

        topology_units = sum(len(model["concepts"]) for model in attention_topology["models"].values())
        topology_units += sum(len(model["concepts"]) for model in relation_topology["models"].values())

        successor_units = int(successor_inventory["headline_metrics"]["num_chains"])
        successor_units *= int(successor_inventory["headline_metrics"]["num_temporal_stages"])

        lift_units = len(category_bridge["family_rows"]) + len(abstraction_ladder["abstract_word_rows"])
        lift_units += 2

        buckets = {
            "specific_dense_signature": HarvestBucket(
                name="specific_dense_signature",
                target_units=specific_units,
                target_kind="concept-specific neuron signatures",
                priority="highest",
                rationale="直接决定 concept-level 编码细节，而且当前 final theorem closure 仍然受限于 specific-bearing exact units 不足。",
            ),
            "protocol_dense_signature": HarvestBucket(
                name="protocol_dense_signature",
                target_units=protocol_units,
                target_kind="protocol-field neuron signatures",
                priority="highest",
                rationale="protocol 是当前从 encoding 走向可执行行为的关键桥，必须从 proxy 变成 dense exact coordinates。",
            ),
            "topology_dense_signature": HarvestBucket(
                name="topology_dense_signature",
                target_units=topology_units,
                target_kind="family/topology neuron signatures",
                priority="high",
                rationale="family patch 和 topology 已经较强，dense harvesting 可以把它们从 strong candidate 推向 neuron-level closure。",
            ),
            "successor_dense_signature": HarvestBucket(
                name="successor_dense_signature",
                target_units=successor_units,
                target_kind="successor-chain neuron signatures",
                priority="highest",
                rationale="successor 仍然是最弱恢复项，必须做专门的 dense chain harvesting。",
            ),
            "lift_dense_signature": HarvestBucket(
                name="lift_dense_signature",
                target_units=lift_units,
                target_kind="lift / abstraction neuron signatures",
                priority="medium",
                rationale="lift 是 final theorem closure 的必要项，但当前瓶颈小于 successor 和 protocol。",
            ),
        }
        return cls(buckets=buckets)

    def summary(self) -> Dict[str, object]:
        total_targets = sum(bucket.target_units for bucket in self.buckets.values())
        highest_priority_targets = sum(
            bucket.target_units for bucket in self.buckets.values() if bucket.priority == "highest"
        )
        return {
            "bucket_count": len(self.buckets),
            "total_target_units": total_targets,
            "highest_priority_target_units": highest_priority_targets,
            "buckets": {
                name: {
                    "target_units": bucket.target_units,
                    "target_kind": bucket.target_kind,
                    "priority": bucket.priority,
                    "rationale": bucket.rationale,
                }
                for name, bucket in self.buckets.items()
            },
        }
