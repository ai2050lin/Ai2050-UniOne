from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json

from research.gpt5.code.dnn_dense_activation_harvest_queue import DenseActivationHarvestQueue


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class SuccessorDenseExportContractRow:
    run_id: str
    source_script: str
    current_exactness_tier: str
    source_artifact: str
    available_axes: List[str]
    required_dense_axes: List[str]
    target_tensor_layout: str
    missing_axes: List[str]
    upgrade_ready_score: float
    upgrade_meaning: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "source_script": self.source_script,
            "current_exactness_tier": self.current_exactness_tier,
            "source_artifact": self.source_artifact,
            "available_axes": self.available_axes,
            "required_dense_axes": self.required_dense_axes,
            "target_tensor_layout": self.target_tensor_layout,
            "missing_axes": self.missing_axes,
            "upgrade_ready_score": self.upgrade_ready_score,
            "upgrade_meaning": self.upgrade_meaning,
        }


@dataclass
class SuccessorDenseExportContract:
    rows: List[SuccessorDenseExportContractRow]

    @classmethod
    def from_repo(cls, root: Path) -> "SuccessorDenseExportContract":
        queue = DenseActivationHarvestQueue.from_pipeline(root)
        successor_rows = [row for row in queue.runs if row.bucket_name == "successor_dense_signature"]
        temp = root / "tests" / "codex_temp"
        online_recovery = load_json(temp / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
        inventory = load_json(temp / "theory_track_successor_strengthened_reasoning_inventory_20260312.json")

        rows: List[SuccessorDenseExportContractRow] = []
        for row in successor_rows:
            if row.exactness_tier == "direct_dense":
                available = ["prompt", "chain", "layer", "neuron", "gate_activation"]
                required = ["prompt", "chain", "stage", "layer", "neuron", "gate_activation"]
                missing = ["stage"]
                score = 0.86
                meaning = "直接 hook 已触到 dense gate activation，当前主要缺失的是显式 stage 轴对齐。"
                artifact = "none_direct_script_path"
                layout = "[num_prompts, num_chains, num_stages, num_layers, d_ff]"
            elif row.exactness_tier == "summary_proxy":
                model_count = len(online_recovery["models"])
                step_count = max(len(model["step_rows"]) for model in online_recovery["models"].values())
                available = ["model", "episode_count", "step", "trigger_rate", "recovery_success_rate", "system_success_rate"]
                required = ["model", "episode", "step", "layer", "head_or_neuron", "activation_tensor", "rollback_mask"]
                missing = ["episode", "layer", "head_or_neuron", "activation_tensor", "rollback_mask"]
                score = min(1.0, 0.20 + 0.10 * model_count + 0.08 * step_count)
                meaning = "online recovery 已有 step 级事件结构，但缺少逐 episode、逐层、逐单元的 dense activation 导出。"
                artifact = "tests/codex_temp/qwen3_deepseek7b_online_recovery_chain_20260310.json"
                layout = "[num_models, num_episodes, num_steps, num_layers, hidden_or_head_dim]"
            else:
                stage_count = int(inventory["headline_metrics"]["num_temporal_stages"])
                chain_count = int(inventory["headline_metrics"]["num_chains"])
                available = ["concept_count", "context_count", "relation_count", "temporal_stage_count", "chain_count", "ratio_metrics"]
                required = ["family", "chain", "stage", "context", "relation", "layer", "neuron", "activation_tensor"]
                missing = ["family", "chain_row_state", "stage_row_state", "layer", "neuron", "activation_tensor"]
                score = min(1.0, 0.18 + 0.004 * stage_count + 0.004 * chain_count)
                meaning = "inventory 已给出 successor 任务空间，但仍缺少每条 chain、每个 stage 的真实 dense row-state。"
                artifact = "tests/codex_temp/theory_track_successor_strengthened_reasoning_inventory_20260312.json"
                layout = "[num_families, num_chains, num_stages, num_layers, d_ff]"

            rows.append(
                SuccessorDenseExportContractRow(
                    run_id=row.run_id,
                    source_script=row.script_path,
                    current_exactness_tier=row.exactness_tier,
                    source_artifact=artifact,
                    available_axes=available,
                    required_dense_axes=required,
                    target_tensor_layout=layout,
                    missing_axes=missing,
                    upgrade_ready_score=float(score),
                    upgrade_meaning=meaning,
                )
            )
        return cls(rows=rows)

    def summary(self) -> Dict[str, object]:
        total = len(self.rows)
        direct_rows = [row for row in self.rows if row.current_exactness_tier == "direct_dense"]
        proxy_rows = [row for row in self.rows if row.current_exactness_tier != "direct_dense"]
        mean_upgrade_ready = sum(row.upgrade_ready_score for row in self.rows) / max(1, total)
        proxy_mean_upgrade_ready = sum(row.upgrade_ready_score for row in proxy_rows) / max(1, len(proxy_rows))
        fully_specified_proxy_rows = sum(1 for row in proxy_rows if len(row.required_dense_axes) >= 6)
        return {
            "successor_rows": total,
            "direct_dense_rows": len(direct_rows),
            "proxy_rows": len(proxy_rows),
            "mean_upgrade_ready_score": float(mean_upgrade_ready),
            "proxy_mean_upgrade_ready_score": float(proxy_mean_upgrade_ready),
            "fully_specified_proxy_rows": fully_specified_proxy_rows,
            "rows": [row.to_dict() for row in self.rows],
        }
