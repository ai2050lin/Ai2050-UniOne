from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class DnnLocalExtractionCompletionBoard:
    root: Path
    clean_bundle: Dict[str, Any]
    clean_runner: Dict[str, Any]
    raw_bundle: Dict[str, Any]
    model_scope: Dict[str, Any]

    @classmethod
    def from_repo(cls, root: Path) -> "DnnLocalExtractionCompletionBoard":
        temp = root / "tests" / "codex_temp"
        return cls(
            root=root,
            clean_bundle=load_json(temp / "dnn_clean_english_execution_bundle_block_20260316.json"),
            clean_runner=load_json(temp / "dnn_clean_execution_runner_block_20260316.json"),
            raw_bundle=load_json(temp / "dnn_1000plus_dense_execution_bundle_block_20260316.json"),
            model_scope=load_json(temp / "dnn_1000plus_model_scope_manifest_block_20260316.json"),
        )

    def summary(self) -> Dict[str, Any]:
        clean_metrics = self.clean_bundle["headline_metrics"]
        clean_runner = self.clean_runner["headline_metrics"]
        raw_metrics = self.raw_bundle["headline_metrics"]
        model_id = self.model_scope["current_execution"]["model_id"]

        offline_preparation_score = min(
            1.0,
            0.25 * min(1.0, clean_metrics["clean_unique_english_nouns"] / 518.0)
            + 0.25 * min(1.0, raw_metrics["total_batched_nouns"] / 1045.0)
            + 0.20 * min(1.0, clean_metrics["clean_batch_count"] / 5.0)
            + 0.15 * min(1.0, raw_metrics["batch_count"] / 9.0)
            + 0.15 * min(1.0, self.model_scope["current_execution"]["launchable_batch_count"] / 9.0),
        )
        real_harvest_completion = float(clean_runner["current_completion_ratio"])
        local_completion_ceiling = 0.72 if real_harvest_completion == 0.0 else min(1.0, 0.72 + 0.28 * real_harvest_completion)

        metric_lines_cn = [
            f"（离线准备完成度）offline_preparation_score = {offline_preparation_score:.4f}",
            f"（真实实采完成度）real_harvest_completion = {real_harvest_completion:.4f}",
            f"（本地条件完成上限）local_completion_ceiling = {local_completion_ceiling:.4f}",
            f"（当前执行模型）current_execution_model = {model_id}",
            f"（干净英文唯一名词数）clean_unique_english_nouns = {clean_metrics['clean_unique_english_nouns']:.4f}",
            f"（原始1000+分批总数）raw_batch_count = {raw_metrics['batch_count']:.4f}",
        ]

        return {
            "headline_metrics": {
                "offline_preparation_score": offline_preparation_score,
                "real_harvest_completion": real_harvest_completion,
                "local_completion_ceiling": local_completion_ceiling,
                "current_execution_model": model_id,
                "clean_unique_english_nouns": clean_metrics["clean_unique_english_nouns"],
                "raw_batch_count": raw_metrics["batch_count"],
                "metric_lines_cn": metric_lines_cn,
            },
            "strict_conclusion": {
                "core_answer": "Under the current local constraints, the project has essentially finished the offline extraction-preparation work: source aggregation, clean-source repair, batch partition, schema wiring, runner tracking, and model-scope clarification. The hard blocker is no longer structure, but missing local model availability for real harvest.",
                "main_hard_gaps": [
                    "real harvest completion is still zero because the DeepSeek model is not available in the local cache",
                    "raw bilingual 1000+ source still carries mojibake on the Chinese side",
                    "successor exact closure remains untouched by the noun-source preparation work alone",
                ],
            },
        }
