from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class DenseRealUnit:
    source: str
    unit_type: str
    weight: int
    carries_macro: bool
    carries_specific: bool


@dataclass
class DenseRealUnitCorpus:
    units: List[DenseRealUnit]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DenseRealUnitCorpus":
        temp = root / "tests" / "codex_temp"
        structure = load_json(temp / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
        recovery = load_json(temp / "qwen3_deepseek7b_real_model_recovery_proxy_atlas_20260310.json")
        mechanism = load_json(temp / "qwen3_deepseek7b_mechanism_bridge_20260309.json")
        codebook = load_json(temp / "concept_family_unified_codebook_20260308.json")
        protocol_field = load_json(temp / "qwen3_deepseek7b_protocol_field_boundary_atlas_20260309.json")
        relation_boundary = load_json(temp / "qwen3_deepseek7b_relation_boundary_atlas_20260309.json")
        relation_topology = load_json(temp / "qwen3_deepseek7b_relation_topology_atlas_20260309.json")
        attention_topology = load_json(temp / "qwen3_deepseek7b_attention_topology_atlas_20260309.json")
        structure_bridge = load_json(temp / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json")
        relation_bridge = load_json(temp / "qwen3_deepseek7b_relation_behavior_bridge_20260309.json")
        concept_protocol = load_json(temp / "qwen3_deepseek7b_concept_protocol_field_mapping_20260309.json")

        units: List[DenseRealUnit] = []

        for model_key, model_payload in structure["models"].items():
            for _row in model_payload["layer_atlas"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="layer_row",
                        weight=1,
                        carries_macro=False,
                        carries_specific=False,
                    )
                )
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="global_summary",
                    weight=4,
                    carries_macro=True,
                    carries_specific=False,
                )
            )

        for model_key, model_payload in recovery["models"].items():
            for _row in model_payload["relation_recovery_rows"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="relation_recovery_row",
                        weight=2,
                        carries_macro=True,
                        carries_specific=False,
                    )
                )
            for _row in model_payload["target_band_rows"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="target_band_row",
                        weight=2,
                        carries_macro=True,
                        carries_specific=False,
                    )
                )
            for _row in model_payload["top_structure_tasks"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="structure_task_row",
                        weight=2,
                        carries_macro=True,
                        carries_specific=True,
                    )
                )
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="recovery_global_summary",
                    weight=4,
                    carries_macro=True,
                    carries_specific=False,
                )
            )

        for model_key in mechanism["models"]:
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="mechanism_bridge_components",
                    weight=6,
                    carries_macro=True,
                    carries_specific=False,
                )
            )

        for model_key, model_payload in protocol_field["models"].items():
            for _concept_name in model_payload["concepts"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="protocol_field_concept",
                        weight=2,
                        carries_macro=True,
                        carries_specific=True,
                    )
                )
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="protocol_field_summary",
                    weight=4,
                    carries_macro=True,
                    carries_specific=False,
                )
            )

        for model_key, model_payload in relation_boundary["models"].items():
            for _relation_name in model_payload["relations"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="relation_boundary_relation",
                        weight=2,
                        carries_macro=True,
                        carries_specific=False,
                    )
                )
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="relation_boundary_summary",
                    weight=3,
                    carries_macro=True,
                    carries_specific=False,
                )
            )

        for model_key, model_payload in relation_topology["models"].items():
            for _concept_name in model_payload["concepts"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="relation_topology_concept",
                        weight=2,
                        carries_macro=True,
                        carries_specific=True,
                    )
                )
            for _family_name in model_payload["family_summary"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="relation_topology_family",
                        weight=1,
                        carries_macro=True,
                        carries_specific=False,
                    )
                )
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="relation_topology_summary",
                    weight=4,
                    carries_macro=True,
                    carries_specific=False,
                )
            )

        for model_key, model_payload in attention_topology["models"].items():
            for _concept_name in model_payload["concepts"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="attention_topology_concept",
                        weight=2,
                        carries_macro=False,
                        carries_specific=True,
                    )
                )
            for _family_name in model_payload["family_summary"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="attention_topology_family",
                        weight=1,
                        carries_macro=False,
                        carries_specific=False,
                    )
                )
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="attention_topology_summary",
                    weight=3,
                    carries_macro=False,
                    carries_specific=False,
                )
            )

        for model_key, model_payload in structure_bridge["models"].items():
            for _task_name in model_payload["tasks"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="structure_bridge_task",
                        weight=2,
                        carries_macro=True,
                        carries_specific=True,
                    )
                )
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="structure_bridge_summary",
                    weight=4,
                    carries_macro=True,
                    carries_specific=False,
                )
            )

        for model_key, model_payload in relation_bridge["models"].items():
            for _relation_name in model_payload["relations"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="relation_bridge_relation",
                        weight=2,
                        carries_macro=True,
                        carries_specific=False,
                    )
                )
            units.append(
                DenseRealUnit(
                    source=model_key,
                    unit_type="relation_bridge_summary",
                    weight=4,
                    carries_macro=True,
                    carries_specific=False,
                )
            )

        for model_key, model_payload in concept_protocol["models"].items():
            for _concept_name in model_payload["concepts"]:
                units.append(
                    DenseRealUnit(
                        source=model_key,
                        unit_type="concept_protocol_concept",
                        weight=3,
                        carries_macro=True,
                        carries_specific=True,
                    )
                )

        for family_name in codebook["family_stats"]:
            units.append(
                DenseRealUnit(
                    source="codebook",
                    unit_type=f"family_stats:{family_name}",
                    weight=2,
                    carries_macro=False,
                    carries_specific=False,
                )
            )
            family_stats = codebook["family_stats"][family_name]
            for _member_name in family_stats["members"]:
                units.append(
                    DenseRealUnit(
                        source="codebook",
                        unit_type=f"group_member:{family_name}",
                        weight=1,
                        carries_macro=False,
                        carries_specific=True,
                    )
                )
        for _pair_name in codebook["pairwise_families"]:
            units.append(
                DenseRealUnit(
                    source="codebook",
                    unit_type="pairwise_family_bridge",
                    weight=2,
                    carries_macro=True,
                    carries_specific=False,
                )
            )
        for concept_name in codebook["spotlight_concepts"]:
            units.append(
                DenseRealUnit(
                    source="codebook",
                    unit_type=f"spotlight:{concept_name}",
                    weight=3,
                    carries_macro=False,
                    carries_specific=True,
                )
            )

        return cls(units=units)

    def summary(self) -> Dict[str, object]:
        total_weight = sum(unit.weight for unit in self.units)
        macro_weight = sum(unit.weight for unit in self.units if unit.carries_macro)
        specific_weight = sum(unit.weight for unit in self.units if unit.carries_specific)
        by_type: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        for unit in self.units:
            by_type[unit.unit_type] = by_type.get(unit.unit_type, 0) + unit.weight
            by_source[unit.source] = by_source.get(unit.source, 0) + unit.weight
        return {
            "unit_count": len(self.units),
            "weighted_units": total_weight,
            "macro_weight": macro_weight,
            "specific_weight": specific_weight,
            "by_type": by_type,
            "by_source": by_source,
        }
