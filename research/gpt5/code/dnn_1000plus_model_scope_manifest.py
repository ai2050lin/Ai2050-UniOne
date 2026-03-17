from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import re


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Dnn1000PlusModelScopeManifest:
    root: Path
    bundle: Dict[str, Any]
    dense_pipeline_source: str

    @classmethod
    def from_repo(cls, root: Path) -> "Dnn1000PlusModelScopeManifest":
        temp = root / "tests" / "codex_temp"
        bundle = load_json(temp / "dnn_1000plus_dense_execution_bundle_block_20260316.json")
        scan_script = root / "tests" / "codex" / "deepseek7b_mass_noun_encoding_scan.py"
        dense_pipeline_source = scan_script.read_text(encoding="utf-8")
        return cls(root=root, bundle=bundle, dense_pipeline_source=dense_pipeline_source)

    def summary(self) -> Dict[str, Any]:
        m = re.search(r'add_argument\("--model-id",\s*default="([^"]+)"\)', self.dense_pipeline_source)
        current_model_id = m.group(1) if m else "unknown"
        task_rows = self.bundle["task_rows"]
        launch_commands = [row["launch_command"] for row in task_rows[:3]]

        metric_lines_cn = [
            f"（当前1000+执行模型）current_execution_model = {current_model_id}",
            "（当前1000+执行入口）current_execution_entry = tests/codex/deepseek7b_mass_noun_encoding_scan.py",
            "（当前1000+主执行链）current_primary_axis = specific_dense_signature",
            "（当前多模型分析补充）current_aux_models = qwen3_4b + deepseek_7b",
            f"（当前可直接启动批次数）launchable_batch_count = {len(task_rows):.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "current_execution": {
                "model_id": current_model_id,
                "entry_script": "tests/codex/deepseek7b_mass_noun_encoding_scan.py",
                "primary_axis": "specific_dense_signature",
                "launchable_batch_count": len(task_rows),
                "sample_launch_commands": launch_commands,
            },
            "analysis_scope": {
                "current_batch_execution_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "historical_structure_analysis_models": ["qwen3_4b", "deepseek_7b"],
                "strict_answer": "当前这条 1000+ 名词批处理主执行链，实际准备运行的是 DeepSeek-R1-Distill-Qwen-7B；Qwen3-4B 目前主要还在结构分析、协议场、关系拓扑和跨模型对照层，不在这条批处理主执行链里。",
            },
            "strict_conclusion": {
                "core_answer": "The current 1000+ noun execution path is a DeepSeek-7B path. Qwen3-4B still exists in the broader research stack, but not as the active primary model for the new batched mass-noun execution bundle.",
                "main_hard_gaps": [
                    "the current 1000+ execution chain is still model-single on the active path",
                    "Qwen-side dense specific execution has not yet been wired into the same batch bundle",
                    "cross-model exact closure still needs to be rebuilt after the first 1000+ DeepSeek harvest wave",
                ],
            },
        }
