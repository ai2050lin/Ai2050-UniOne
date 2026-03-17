from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class DenseExportSchemaSpec:
    bucket_name: str
    required_axes: List[str]
    tensor_field: str
    target_shape_hint: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_name": self.bucket_name,
            "required_axes": self.required_axes,
            "tensor_field": self.tensor_field,
            "target_shape_hint": self.target_shape_hint,
            "rationale": self.rationale,
        }


@dataclass
class DnnJointDenseExportSchema:
    sprint_manifest: Dict[str, Any]
    queue_block: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnJointDenseExportSchema":
        temp = root / "tests" / "codex_temp"
        return cls(
            sprint_manifest=load_json(temp / "dnn_joint_closure_sprint_manifest_block_20260315.json"),
            queue_block=load_json(temp / "dnn_dense_activation_harvest_queue_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        queue_rows = self.queue_block["queue_summary"]["runs"]
        sprint_axes = self.sprint_manifest["headline_metrics"]["sprint_axes"]

        specs = [
            DenseExportSchemaSpec(
                bucket_name="specific_dense_signature",
                required_axes=["run_id", "model_scope", "concept_group", "prompt_id", "layer", "neuron", "concept", "category"],
                tensor_field="activation_tensor",
                target_shape_hint="[num_prompts, num_layers, d_ff] or flattened [num_prompts, num_layers*d_ff]",
                rationale="specific 线要能回到 family -> specific 的 exact neuron signatures。",
            ),
            DenseExportSchemaSpec(
                bucket_name="protocol_dense_signature",
                required_axes=["run_id", "model_scope", "concept_group", "prompt_id", "layer", "head_or_neuron", "protocol_field"],
                tensor_field="activation_tensor",
                target_shape_hint="[num_prompts, num_layers, num_heads] with protocol-field margins",
                rationale="protocol 线需要作为 dense exact evidence 的统一底座。",
            ),
            DenseExportSchemaSpec(
                bucket_name="successor_dense_signature",
                required_axes=["run_id", "model_scope", "chain", "stage", "layer", "head_or_neuron", "context", "relation"],
                tensor_field="activation_tensor",
                target_shape_hint="[num_chains, num_stages, num_layers, d_ff]",
                rationale="successor 线必须显式保留 chain-stage 结构，否则 exact closure 无法抬升。",
            ),
        ]

        available_buckets = {row["bucket_name"] for row in queue_rows}
        launchable_buckets = {row["bucket_name"] for row in queue_rows if row["launchable"]}
        schema_rows = []
        for spec in specs:
            schema_rows.append({
                **spec.to_dict(),
                "queue_present": spec.bucket_name in available_buckets,
                "launchable": spec.bucket_name in launchable_buckets,
            })

        ready_count = sum(1 for row in schema_rows if row["launchable"])
        schema_ready_score = (
            0.40 * (ready_count / max(1, len(schema_rows)))
            + 0.30 * min(1.0, len(sprint_axes) / 3.0)
            + 0.30
        )

        metric_lines_cn = [
            f"（联合dense导出schema准备度）schema_ready_score = {schema_ready_score:.4f}",
            f"（schema覆盖桶数）schema_bucket_count = {len(schema_rows)}",
            f"（可直接启动schema桶数）launchable_schema_bucket_count = {ready_count}",
            f"（冲刺主轴数）sprint_axis_count = {len(sprint_axes)}",
        ]

        return {
            "schema_ready_score": float(schema_ready_score),
            "schema_rows": schema_rows,
            "metric_lines_cn": metric_lines_cn,
            "critical_path": [
                "specific_dense_signature 按统一 neuron-level tensor schema 落盘",
                "protocol_dense_signature 按统一 field/head tensor schema 落盘",
                "successor_dense_signature 按统一 chain-stage-layer tensor schema 落盘",
            ],
        }
