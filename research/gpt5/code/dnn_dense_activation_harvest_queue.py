from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from research.gpt5.code.dnn_dense_activation_harvest_pipeline import DenseActivationHarvestPipeline


@dataclass
class HarvestRunSpec:
    run_id: str
    bucket_name: str
    priority: str
    script_path: str
    model_scope: str
    concept_group: str
    prompt_family: str
    capture_site: str
    tensor_layout: str
    exactness_tier: str
    supports_direct_dense: bool
    target_view: str
    launchable: bool

    def to_dict(self, root: Path) -> Dict[str, object]:
        path = root / self.script_path
        return {
            "run_id": self.run_id,
            "bucket_name": self.bucket_name,
            "priority": self.priority,
            "script_path": self.script_path,
            "script_exists": path.exists(),
            "model_scope": self.model_scope,
            "concept_group": self.concept_group,
            "prompt_family": self.prompt_family,
            "capture_site": self.capture_site,
            "tensor_layout": self.tensor_layout,
            "exactness_tier": self.exactness_tier,
            "supports_direct_dense": self.supports_direct_dense,
            "target_view": self.target_view,
            "launchable": self.launchable,
        }


@dataclass
class DenseActivationHarvestQueue:
    runs: List[HarvestRunSpec]

    @classmethod
    def from_pipeline(cls, root: Path) -> "DenseActivationHarvestQueue":
        pipeline = DenseActivationHarvestPipeline.from_repo(root)
        selected = [
            pipeline.tasks["specific_dense_signature"],
            pipeline.tasks["protocol_dense_signature"],
            pipeline.tasks["successor_dense_signature"],
        ]

        runs: List[HarvestRunSpec] = []
        for task in selected:
            for idx, script in enumerate(task.scripts):
                concept_group = task.concept_groups[min(idx, len(task.concept_groups) - 1)]
                prompt_family = task.prompt_families[min(idx, len(task.prompt_families) - 1)]
                capture_site = task.capture_sites[min(idx, len(task.capture_sites) - 1)]
                launchable = bool(script.supports_direct_dense and (root / script.relative_path).exists())
                if task.bucket_name == "protocol_dense_signature" and script.exactness_tier == "head_dense":
                    launchable = bool((root / script.relative_path).exists())
                run_id = f"{task.bucket_name}__{idx+1}"
                runs.append(
                    HarvestRunSpec(
                        run_id=run_id,
                        bucket_name=task.bucket_name,
                        priority=task.priority,
                        script_path=script.relative_path,
                        model_scope=script.model_scope,
                        concept_group=concept_group,
                        prompt_family=prompt_family,
                        capture_site=capture_site,
                        tensor_layout=task.tensor_layout,
                        exactness_tier=script.exactness_tier,
                        supports_direct_dense=script.supports_direct_dense,
                        target_view=script.target_view,
                        launchable=launchable,
                    )
                )
        return cls(runs=runs)

    def summary(self, root: Path) -> Dict[str, object]:
        run_rows = [run.to_dict(root) for run in self.runs]
        total_runs = len(run_rows)
        launchable_runs = sum(1 for row in run_rows if row["launchable"])
        highest_priority_runs = sum(1 for row in run_rows if row["priority"] == "highest")
        direct_dense_runs = sum(1 for row in run_rows if row["exactness_tier"] == "direct_dense")
        head_dense_runs = sum(1 for row in run_rows if row["exactness_tier"] == "head_dense")
        summary_proxy_runs = sum(1 for row in run_rows if row["exactness_tier"] == "summary_proxy")
        inventory_proxy_runs = sum(1 for row in run_rows if row["exactness_tier"] == "inventory_proxy")
        exact_dense_runs = direct_dense_runs + head_dense_runs
        queue_ready_score = min(
            1.0,
            0.35 * min(1.0, launchable_runs / max(1, total_runs))
            + 0.25 * min(1.0, exact_dense_runs / max(1, total_runs))
            + 0.20 * min(1.0, highest_priority_runs / 10.0)
            + 0.20,
        )
        return {
            "total_runs": total_runs,
            "launchable_runs": launchable_runs,
            "highest_priority_runs": highest_priority_runs,
            "direct_dense_runs": direct_dense_runs,
            "head_dense_runs": head_dense_runs,
            "summary_proxy_runs": summary_proxy_runs,
            "inventory_proxy_runs": inventory_proxy_runs,
            "queue_ready_score": float(queue_ready_score),
            "runs": run_rows,
        }

    def successor_exactness_summary(self, root: Path) -> Dict[str, object]:
        rows = [run.to_dict(root) for run in self.runs if run.bucket_name == "successor_dense_signature"]
        total = len(rows)
        direct_dense = sum(1 for row in rows if row["exactness_tier"] == "direct_dense")
        summary_proxy = sum(1 for row in rows if row["exactness_tier"] == "summary_proxy")
        inventory_proxy = sum(1 for row in rows if row["exactness_tier"] == "inventory_proxy")
        exact_dense_ratio = float(direct_dense / max(1, total))
        proxy_ratio = float((summary_proxy + inventory_proxy) / max(1, total))
        dense_exact_closure = bool(total > 0 and exact_dense_ratio >= 0.80 and proxy_ratio <= 0.20)
        return {
            "successor_run_count": total,
            "direct_dense_run_count": direct_dense,
            "summary_proxy_run_count": summary_proxy,
            "inventory_proxy_run_count": inventory_proxy,
            "direct_dense_ratio": exact_dense_ratio,
            "proxy_ratio": proxy_ratio,
            "dense_exact_closure": dense_exact_closure,
            "runs": rows,
        }
