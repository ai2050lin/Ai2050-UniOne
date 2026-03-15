from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class SuccessorStageRow:
    source_name: str
    evidence_tier: str
    model_scope: str
    stage_name: str
    chain_count: int
    trigger_rate: float
    recovery_success_rate: float
    exactness_weight: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_name": self.source_name,
            "evidence_tier": self.evidence_tier,
            "model_scope": self.model_scope,
            "stage_name": self.stage_name,
            "chain_count": self.chain_count,
            "trigger_rate": self.trigger_rate,
            "recovery_success_rate": self.recovery_success_rate,
            "exactness_weight": self.exactness_weight,
        }


@dataclass
class DnnSuccessorStageRowCorpus:
    rows: List[SuccessorStageRow]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnSuccessorStageRowCorpus":
        temp = root / "tests" / "codex_temp"
        online_recovery = load_json(temp / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
        inventory = load_json(temp / "theory_track_successor_strengthened_reasoning_inventory_20260312.json")

        rows: List[SuccessorStageRow] = []
        for model_name, model_payload in online_recovery["models"].items():
            for row in model_payload["step_rows"]:
                rows.append(
                    SuccessorStageRow(
                        source_name="online_recovery_chain",
                        evidence_tier="summary_proxy",
                        model_scope=model_name,
                        stage_name=str(row["step"]),
                        chain_count=int(model_payload["episode_count"]),
                        trigger_rate=float(row["trigger_rate"]),
                        recovery_success_rate=float(row["recovery_success_rate"]),
                        exactness_weight=0.35,
                    )
                )

        num_stages = int(inventory["headline_metrics"]["num_temporal_stages"])
        num_chains = int(inventory["headline_metrics"]["num_chains"])
        successor_ratio = float(inventory["headline_metrics"]["chain_successor_to_cross_stage_ratio"])
        trigger_template = max(0.0, min(1.0, 1.0 - successor_ratio))
        recovery_template = max(0.0, min(1.0, successor_ratio))
        for idx in range(num_stages):
            rows.append(
                SuccessorStageRow(
                    source_name="successor_inventory",
                    evidence_tier="inventory_proxy",
                    model_scope="inventory_space",
                    stage_name=f"t{idx}",
                    chain_count=num_chains,
                    trigger_rate=trigger_template,
                    recovery_success_rate=recovery_template,
                    exactness_weight=0.25,
                )
            )
        return cls(rows=rows)

    def summary(self) -> Dict[str, object]:
        online_rows = [row for row in self.rows if row.source_name == "online_recovery_chain"]
        inventory_rows = [row for row in self.rows if row.source_name == "successor_inventory"]
        weighted_exact_rows = sum(row.exactness_weight for row in self.rows)
        return {
            "stage_row_count": len(self.rows),
            "online_recovery_stage_rows": len(online_rows),
            "inventory_stage_rows": len(inventory_rows),
            "mean_trigger_rate": float(sum(row.trigger_rate for row in self.rows) / max(1, len(self.rows))),
            "mean_recovery_success_rate": float(
                sum(row.recovery_success_rate for row in self.rows) / max(1, len(self.rows))
            ),
            "weighted_exact_stage_rows": float(weighted_exact_rows),
            "rows": [row.to_dict() for row in self.rows],
        }
