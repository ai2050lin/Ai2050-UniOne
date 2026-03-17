from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class DnnJointClosureSprintManifest:
    leverage_board: Dict[str, Any]
    queue_board: Dict[str, Any]
    joint_board: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnJointClosureSprintManifest":
        temp = root / "tests" / "codex_temp"
        return cls(
            leverage_board=load_json(temp / "dnn_joint_closure_leverage_board_block_20260315.json"),
            queue_board=load_json(temp / "dnn_dense_activation_harvest_pipeline_block_20260315.json"),
            joint_board=load_json(temp / "dnn_joint_exact_closure_board_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        leverage = self.leverage_board["headline_metrics"]
        pipeline_metrics = self.queue_board["headline_metrics"]
        pipeline_tasks = self.queue_board["pipeline_summary"]["tasks"]
        joint = self.joint_board["headline_metrics"]

        sprint_axes = [
            {
                "axis": "family-to-specific exact closure",
                "priority_rank": 1,
                "why": "当前单块杠杆最高，直接决定 concept-specific 是否能从 family-level 走向 exact closure。",
                "recommended_bucket": "specific_dense_signature",
                "task_readiness": float(pipeline_tasks["specific_dense_signature"]["task_readiness"]),
                "launch_ready": bool(pipeline_tasks["specific_dense_signature"]["launch_ready"]),
                "target_units": int(pipeline_tasks["specific_dense_signature"]["target_units"]),
            },
            {
                "axis": "successor exact closure",
                "priority_rank": 2,
                "why": "当前双块最佳组合必须包含 successor，否则系统编码无法进入连续后继和语言链路。",
                "recommended_bucket": "successor_dense_signature",
                "task_readiness": float(pipeline_tasks["successor_dense_signature"]["task_readiness"]),
                "launch_ready": bool(pipeline_tasks["successor_dense_signature"]["launch_ready"]),
                "target_units": int(pipeline_tasks["successor_dense_signature"]["target_units"]),
            },
            {
                "axis": "dense exact evidence",
                "priority_rank": 3,
                "why": "虽然单块杠杆略低，但它是三块联动中的底座，不提升会持续压制 theorem-grade closure。",
                "recommended_bucket": "protocol_dense_signature",
                "task_readiness": float(pipeline_tasks["protocol_dense_signature"]["task_readiness"]),
                "launch_ready": bool(pipeline_tasks["protocol_dense_signature"]["launch_ready"]),
                "target_units": int(pipeline_tasks["protocol_dense_signature"]["target_units"]),
            },
        ]

        sprint_readiness = (
            0.30 * sprint_axes[0]["task_readiness"]
            + 0.30 * sprint_axes[1]["task_readiness"]
            + 0.20 * sprint_axes[2]["task_readiness"]
            + 0.20 * float(pipeline_metrics["pipeline_ready_score"])
        )

        coupled_gain_ceiling = float(leverage["best_all_scenario"]["delta_coupled"])
        baseline_coupled = float(joint["coupled_exact_closure_score"])
        projected_stage_target = baseline_coupled + coupled_gain_ceiling

        metric_lines_cn = [
            f"（联合冲刺准备度）sprint_readiness_score = {sprint_readiness:.4f}",
            f"（当前联合闭合基线）baseline_coupled_exact_closure = {baseline_coupled:.4f}",
            f"（本轮联合提升上限）coupled_gain_ceiling = {coupled_gain_ceiling:.4f}",
            f"（阶段目标联合闭合）projected_stage_target = {projected_stage_target:.4f}",
            f"（可直接启动的高优先级桶数）launchable_high_priority_buckets = {sum(1 for axis in sprint_axes if axis['launch_ready'])}",
        ]

        return {
            "sprint_readiness_score": float(sprint_readiness),
            "baseline_coupled_exact_closure": baseline_coupled,
            "coupled_gain_ceiling": coupled_gain_ceiling,
            "projected_stage_target": float(projected_stage_target),
            "sprint_axes": sprint_axes,
            "metric_lines_cn": metric_lines_cn,
            "critical_path": [
                "先启动 specific_dense_signature，围绕 family-to-specific exact closure 形成第一杠杆主轴",
                "并行启动 successor_dense_signature，把第二杠杆主轴直接接入链路级 dense export",
                "以 protocol_dense_signature 作为 dense exact evidence 底座，防止联合闭合继续被证据上限锁死",
            ],
        }
