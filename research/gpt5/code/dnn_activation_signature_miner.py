from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ActivationSignature:
    concept: str
    source: str
    family: str
    specific_dim_count: int
    specific_layer_count: int
    protocol_margin: float
    topology_margin: float
    boundary_causal_margin: float
    carries_protocol: bool
    carries_topology: bool
    carries_specific: bool


@dataclass
class ActivationSignatureMiner:
    signatures: List[ActivationSignature]

    @classmethod
    def from_artifacts(cls, root: Path) -> "ActivationSignatureMiner":
        temp = root / "tests" / "codex_temp"
        codebook = load_json(temp / "concept_family_unified_codebook_20260308.json")
        protocol_map = load_json(temp / "qwen3_deepseek7b_concept_protocol_field_mapping_20260309.json")
        protocol_boundary = load_json(temp / "qwen3_deepseek7b_protocol_field_boundary_atlas_20260309.json")
        attention_topology = load_json(temp / "qwen3_deepseek7b_attention_topology_atlas_20260309.json")
        relation_topology = load_json(temp / "qwen3_deepseek7b_relation_topology_atlas_20260309.json")

        signatures: List[ActivationSignature] = []

        for family_name, family_row in codebook["family_stats"].items():
            for concept_name, concept_row in family_row["concept_specific"].items():
                signatures.append(
                    ActivationSignature(
                        concept=concept_name,
                        source="codebook",
                        family=family_name,
                        specific_dim_count=len(concept_row["top_specific_dims"]),
                        specific_layer_count=len(concept_row["layer_distribution"]),
                        protocol_margin=0.0,
                        topology_margin=float(family_row["family_cosine_margin"]),
                        boundary_causal_margin=0.0,
                        carries_protocol=False,
                        carries_topology=True,
                        carries_specific=True,
                    )
                )

        for model_key, model_payload in protocol_map["models"].items():
            for concept_name, row in model_payload["concepts"].items():
                summary = row["summary"]
                signatures.append(
                    ActivationSignature(
                        concept=f"{model_key}:{concept_name}",
                        source=model_key,
                        family=str(row["true_field"]),
                        specific_dim_count=0,
                        specific_layer_count=0,
                        protocol_margin=float(summary["margin_vs_second"]),
                        topology_margin=0.0,
                        boundary_causal_margin=0.0,
                        carries_protocol=True,
                        carries_topology=False,
                        carries_specific=True,
                    )
                )

        for model_key, model_payload in protocol_boundary["models"].items():
            for concept_name, row in model_payload["concepts"].items():
                best_causal_margin = max(
                    float(item["summary"]["causal_margin"])
                    for item in row["k_scan"].values()
                )
                signatures.append(
                    ActivationSignature(
                        concept=f"{model_key}:{concept_name}",
                        source=model_key,
                        family=str(row["true_field"]),
                        specific_dim_count=0,
                        specific_layer_count=0,
                        protocol_margin=float(row["baseline_margin"]),
                        topology_margin=0.0,
                        boundary_causal_margin=best_causal_margin,
                        carries_protocol=True,
                        carries_topology=False,
                        carries_specific=True,
                    )
                )

        for model_key, model_payload in attention_topology["models"].items():
            for concept_name, row in model_payload["concepts"].items():
                summary = row["summary"]
                signatures.append(
                    ActivationSignature(
                        concept=f"{model_key}:{concept_name}",
                        source=model_key,
                        family=str(row["true_family"]),
                        specific_dim_count=0,
                        specific_layer_count=0,
                        protocol_margin=0.0,
                        topology_margin=float(summary["margin_vs_best_wrong"]),
                        boundary_causal_margin=0.0,
                        carries_protocol=False,
                        carries_topology=True,
                        carries_specific=True,
                    )
                )

        for model_key, model_payload in relation_topology["models"].items():
            for concept_name, row in model_payload["concepts"].items():
                summary = row["summary"]
                signatures.append(
                    ActivationSignature(
                        concept=f"{model_key}:{concept_name}",
                        source=model_key,
                        family=str(row["true_family"]),
                        specific_dim_count=0,
                        specific_layer_count=0,
                        protocol_margin=0.0,
                        topology_margin=float(summary["margin_vs_best_wrong"]),
                        boundary_causal_margin=0.0,
                        carries_protocol=False,
                        carries_topology=True,
                        carries_specific=True,
                    )
                )

        return cls(signatures=signatures)

    def summary(self) -> Dict[str, object]:
        unique_concepts = sorted({sig.concept for sig in self.signatures})
        sources = sorted({sig.source for sig in self.signatures})
        specific_rows = [sig for sig in self.signatures if sig.carries_specific]
        protocol_rows = [sig for sig in self.signatures if sig.carries_protocol]
        topology_rows = [sig for sig in self.signatures if sig.carries_topology]
        specific_dim_rows = [sig for sig in self.signatures if sig.specific_dim_count > 0]
        mean_specific_dim_count = (
            sum(sig.specific_dim_count for sig in specific_dim_rows) / max(1, len(specific_dim_rows))
        )
        mean_specific_layer_count = (
            sum(sig.specific_layer_count for sig in specific_dim_rows) / max(1, len(specific_dim_rows))
        )
        mean_protocol_margin = (
            sum(sig.protocol_margin for sig in protocol_rows) / max(1, len(protocol_rows))
        )
        mean_topology_margin = (
            sum(sig.topology_margin for sig in topology_rows) / max(1, len(topology_rows))
        )
        mean_boundary_causal_margin = (
            sum(sig.boundary_causal_margin for sig in protocol_rows) / max(1, len(protocol_rows))
        )
        return {
            "signature_rows": len(self.signatures),
            "unique_concepts": len(unique_concepts),
            "sources": sources,
            "specific_signature_rows": len(specific_rows),
            "protocol_signature_rows": len(protocol_rows),
            "topology_signature_rows": len(topology_rows),
            "mean_specific_dim_count": float(mean_specific_dim_count),
            "mean_specific_layer_count": float(mean_specific_layer_count),
            "mean_protocol_margin": float(mean_protocol_margin),
            "mean_topology_margin": float(mean_topology_margin),
            "mean_boundary_causal_margin": float(mean_boundary_causal_margin),
        }

