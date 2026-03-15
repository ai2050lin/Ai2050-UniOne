from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class SuccessorCorpusRow:
    name: str
    evidence_tier: str
    standardized_units: int
    exactness_weight: float
    stage_resolved: bool
    carries_dense_tensor: bool
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "evidence_tier": self.evidence_tier,
            "standardized_units": self.standardized_units,
            "exactness_weight": self.exactness_weight,
            "stage_resolved": self.stage_resolved,
            "carries_dense_tensor": self.carries_dense_tensor,
            "notes": self.notes,
        }


@dataclass
class DnnSuccessorRealCorpus:
    rows: Dict[str, SuccessorCorpusRow]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnSuccessorRealCorpus":
        temp = root / "tests" / "codex_temp"
        extraction = load_json(temp / "dnn_successor_structure_extraction_block_20260315.json")
        online_recovery = load_json(temp / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
        inventory = load_json(temp / "theory_track_successor_strengthened_reasoning_inventory_20260312.json")
        contract = load_json(temp / "dnn_successor_dense_export_contract_block_20260315.json")

        online_model_count = len(online_recovery["models"])
        online_step_rows = sum(len(model["step_rows"]) for model in online_recovery["models"].values())
        online_episode_count = sum(int(model["episode_count"]) for model in online_recovery["models"].values())
        inventory_units = int(inventory["headline_metrics"]["num_chains"]) * int(
            inventory["headline_metrics"]["num_temporal_stages"]
        )
        extracted_units = len(extraction["extracted_components"])
        contract_units = sum(len(row["required_dense_axes"]) for row in contract["contract_summary"]["rows"])

        rows = {
            "direct_multihop_gate_dense": SuccessorCorpusRow(
                name="direct_multihop_gate_dense",
                evidence_tier="direct_dense",
                standardized_units=96,
                exactness_weight=1.0,
                stage_resolved=False,
                carries_dense_tensor=True,
                notes="DeepSeek multi-hop route path already touches real gate activations, but still lacks explicit stage-axis alignment.",
            ),
            "online_recovery_proxy_rows": SuccessorCorpusRow(
                name="online_recovery_proxy_rows",
                evidence_tier="summary_proxy",
                standardized_units=online_step_rows + online_model_count + online_episode_count // 60,
                exactness_weight=0.35,
                stage_resolved=True,
                carries_dense_tensor=False,
                notes="Online recovery has step-level structure and episode counts, but not episode-layer-unit dense tensors.",
            ),
            "successor_inventory_proxy_rows": SuccessorCorpusRow(
                name="successor_inventory_proxy_rows",
                evidence_tier="inventory_proxy",
                standardized_units=inventory_units,
                exactness_weight=0.25,
                stage_resolved=True,
                carries_dense_tensor=False,
                notes="Inventory already spans chain and stage space, but still lacks dense row-state tensors.",
            ),
            "dnn_successor_structure_rows": SuccessorCorpusRow(
                name="dnn_successor_structure_rows",
                evidence_tier="structured_math",
                standardized_units=extracted_units,
                exactness_weight=0.70,
                stage_resolved=False,
                carries_dense_tensor=False,
                notes="DNN-side successor structure extraction gives explicit law components, but they are still candidate math terms rather than dense neuron tensors.",
            ),
            "successor_export_contract_rows": SuccessorCorpusRow(
                name="successor_export_contract_rows",
                evidence_tier="export_contract",
                standardized_units=contract_units,
                exactness_weight=0.40,
                stage_resolved=True,
                carries_dense_tensor=False,
                notes="Export contracts define which dense axes are still missing on each successor path.",
            ),
        }
        return cls(rows=rows)

    def summary(self) -> Dict[str, object]:
        total_units = sum(row.standardized_units for row in self.rows.values())
        exact_dense_units = sum(
            row.standardized_units for row in self.rows.values() if row.evidence_tier == "direct_dense"
        )
        proxy_units = sum(
            row.standardized_units
            for row in self.rows.values()
            if row.evidence_tier in {"summary_proxy", "inventory_proxy"}
        )
        stage_resolved_units = sum(row.standardized_units for row in self.rows.values() if row.stage_resolved)
        dense_tensor_units = sum(row.standardized_units for row in self.rows.values() if row.carries_dense_tensor)
        weighted_exact_units = sum(row.standardized_units * row.exactness_weight for row in self.rows.values())
        exactness_fraction = float(weighted_exact_units / max(1, total_units))
        return {
            "total_successor_units": total_units,
            "exact_dense_units": exact_dense_units,
            "proxy_units": proxy_units,
            "stage_resolved_units": stage_resolved_units,
            "dense_tensor_units": dense_tensor_units,
            "weighted_exact_units": float(weighted_exact_units),
            "exactness_fraction": exactness_fraction,
            "rows": {name: row.to_dict() for name, row in self.rows.items()},
        }
