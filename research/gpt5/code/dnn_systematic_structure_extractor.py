from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json

from research.gpt5.code.dnn_dense_real_unit_corpus import DenseRealUnitCorpus


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ExtractionSourceBlock:
    name: str
    evidence_tier: str
    standardized_units: int
    families: int
    scales: List[str]
    notes: str


@dataclass
class SystematicStructureCorpus:
    source_blocks: List[ExtractionSourceBlock]
    support_metrics: Dict[str, float]
    general_metrics: Dict[str, float]

    @classmethod
    def from_artifacts(cls, root: Path) -> "SystematicStructureCorpus":
        temp = root / "tests" / "codex_temp"

        real_codebook = load_json(temp / "dnn_real_derived_codebook_atlas_block_20260315.json")
        multimodel = load_json(temp / "dnn_multimodel_real_atlas_block_20260315.json")
        unified = load_json(temp / "dnn_unified_parametric_concept_atlas_block_20260315.json")
        large_inventory = load_json(temp / "theory_track_large_scale_concept_inventory_analysis_20260312.json")
        relation_context = load_json(temp / "theory_track_large_inventory_relation_context_synthesis_20260312.json")
        relation_attribute = load_json(temp / "theory_track_concept_relation_attribute_atlas_synthesis_20260312.json")
        triscale = load_json(temp / "dnn_parametric_triscale_encoding_system_block_20260315.json")
        multimodel_specific = load_json(temp / "dnn_multimodel_specific_reconstruction_block_20260315.json")
        structured_operator = load_json(temp / "dnn_multimodel_structured_canonical_operator_block_20260315.json")
        successor = load_json(temp / "dnn_successor_structure_extraction_block_20260315.json")
        dense_real_units = DenseRealUnitCorpus.from_artifacts(root).summary()

        relation_attribute_units = (
            len(relation_attribute["atlas_layers"])
            + len(relation_attribute["representative_entries"])
            + len(relation_attribute["representative_attribute_axes"])
            + len(relation_attribute["relation_templates"])
        )
        inventory_units = int(large_inventory["headline_metrics"]["num_concepts"]) + int(
            relation_context["inventory_scale"]["concept_relation_context_count"]
        )

        source_blocks = [
            ExtractionSourceBlock(
                name="real_codebook_sparse",
                evidence_tier="real_sparse",
                standardized_units=int(real_codebook["headline_metrics"]["total_entries"]),
                families=len(real_codebook["atlas_summary"]["family_counts"]),
                scales=["micro", "meso"],
                notes="真实 codebook 提取，支持 specific/family/shared 与早中晚区域摘要。",
            ),
            ExtractionSourceBlock(
                name="multimodel_real_atlas",
                evidence_tier="real_summary",
                standardized_units=int(multimodel["headline_metrics"]["total_entries"]),
                families=int(multimodel["headline_metrics"]["num_families"]),
                scales=["micro", "meso", "macro"],
                notes="融合 codebook 与 qwen/deepseek 机制条目。",
            ),
            ExtractionSourceBlock(
                name="dense_real_unit_corpus",
                evidence_tier="real_row_level",
                standardized_units=int(dense_real_units["weighted_units"]),
                families=int(multimodel["headline_metrics"]["num_families"]),
                scales=["micro", "meso", "macro"],
                notes="qwen/deepseek layer rows、recovery rows、task rows 与 codebook spotlight 的标准化真实单位。",
            ),
            ExtractionSourceBlock(
                name="unified_parametric_atlas",
                evidence_tier="mixed_with_synthetic",
                standardized_units=int(unified["atlas_summary"]["total_entries"]),
                families=len(unified["atlas_summary"]["family_counts"]),
                scales=["micro", "meso", "macro"],
                notes="统一参数 atlas，但仍混合 exemplar 与 synthetic scale-up。",
            ),
            ExtractionSourceBlock(
                name="large_scale_inventory_mass",
                evidence_tier="inventory_mass",
                standardized_units=inventory_units,
                families=int(large_inventory["headline_metrics"]["num_families"]),
                scales=["meso", "macro"],
                notes="大规模 concept-only 与 relation-context inventory 规模信号。",
            ),
            ExtractionSourceBlock(
                name="relation_attribute_atlas",
                evidence_tier="theory_structured",
                standardized_units=int(relation_attribute_units),
                families=int(large_inventory["headline_metrics"]["num_families"]),
                scales=["micro", "meso", "macro"],
                notes="属性轴、关系模板、atlas layers 的结构化对象。",
            ),
        ]

        support_metrics = {
            "family_fit_strength": float(triscale["headline_metrics"]["family_fit_strength"]),
            "wrong_family_margin": float(triscale["headline_metrics"]["wrong_family_margin"]),
            "regional_reconstructability_score": float(triscale["headline_metrics"]["regional_reconstructability_score"]),
            "inverse_reconstruction_confidence": float(triscale["headline_metrics"]["inverse_reconstruction_confidence"]),
            "contextual_specific_gain": float(
                multimodel_specific["headline_metrics"]["contextual_family_to_specific_gain"]
            ),
            "family_specific_gain": float(multimodel_specific["headline_metrics"]["family_to_specific_gain"]),
            "structured_specific_gain": float(structured_operator["headline_metrics"]["structured_specific_gain"]),
            "structured_macro_gain": float(structured_operator["headline_metrics"]["structured_macro_gain"]),
            "successor_transport_margin": float(successor["headline_metrics"]["transport_margin"]),
            "extracted_successor_score": float(successor["headline_metrics"]["extracted_successor_score"]),
            "cross_to_within_ratio": float(large_inventory["headline_metrics"]["cross_to_within_ratio"]),
            "mean_offset_norm": float(large_inventory["headline_metrics"]["mean_offset_norm"]),
            "dense_real_macro_weight": float(dense_real_units["macro_weight"]),
            "dense_real_specific_weight": float(dense_real_units["specific_weight"]),
        }

        total_units = sum(block.standardized_units for block in source_blocks)
        exact_real_units = sum(
            block.standardized_units
            for block in source_blocks
            if block.evidence_tier in {"real_sparse", "real_summary", "real_row_level"}
        )
        synthetic_units = int(unified["atlas_summary"]["synthetic_entries"])
        scales = sorted({scale for block in source_blocks for scale in block.scales})
        max_families = max(block.families for block in source_blocks)

        general_metrics = {
            "total_standardized_units": float(total_units),
            "exact_real_units": float(exact_real_units),
            "synthetic_units": float(synthetic_units),
            "inventory_mass_units": float(inventory_units),
            "exact_real_fraction": float(exact_real_units / max(total_units, 1)),
            "scale_coverage": float(len(scales) / 3.0),
            "family_coverage": float(min(1.0, max_families / 6.0)),
        }
        return cls(source_blocks=source_blocks, support_metrics=support_metrics, general_metrics=general_metrics)

    def source_summary(self) -> Dict[str, Dict[str, object]]:
        return {
            block.name: {
                "evidence_tier": block.evidence_tier,
                "standardized_units": block.standardized_units,
                "families": block.families,
                "scales": block.scales,
                "notes": block.notes,
            }
            for block in self.source_blocks
        }
