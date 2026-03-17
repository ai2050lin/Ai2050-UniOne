from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class DnnCleanExecutionRunner:
    root: Path
    bundle: Dict[str, Any]

    @classmethod
    def from_repo(cls, root: Path) -> "DnnCleanExecutionRunner":
        temp = root / "tests" / "codex_temp"
        return cls(
            root=root,
            bundle=load_json(temp / "dnn_clean_english_execution_bundle_block_20260316.json"),
        )

    def summary(self) -> Dict[str, Any]:
        batch_rows = self.bundle["batch_rows"]
        run_rows: List[Dict[str, Any]] = []
        completed = 0
        for row in batch_rows:
            out_dir = self.root / row["launch_command"].split("--output-dir ")[-1].strip().strip('"')
            json_path = out_dir / "mass_noun_encoding_scan.json"
            md_path = out_dir / "MASS_NOUN_ENCODING_SCAN_REPORT.md"
            done = json_path.exists() and md_path.exists()
            if done:
                completed += 1
            run_rows.append(
                {
                    "batch_id": row["batch_id"],
                    "noun_count": row["noun_count"],
                    "csv_relative_path": row["csv_relative_path"],
                    "launch_command": row["launch_command"],
                    "output_dir": str(out_dir.relative_to(self.root)).replace("\\", "/"),
                    "json_exists": json_path.exists(),
                    "report_exists": md_path.exists(),
                    "completed": done,
                }
            )

        ready_ratio = completed / max(1, len(run_rows))
        metric_lines_cn = [
            f"（干净执行总批次数）clean_total_batches = {len(run_rows):.4f}",
            f"（已完成批次数）completed_batches = {completed:.4f}",
            f"（未完成批次数）remaining_batches = {len(run_rows) - completed:.4f}",
            f"（当前实采完成比例）current_completion_ratio = {ready_ratio:.4f}",
        ]

        return {
            "headline_metrics": {
                "clean_total_batches": len(run_rows),
                "completed_batches": completed,
                "remaining_batches": len(run_rows) - completed,
                "current_completion_ratio": ready_ratio,
                "metric_lines_cn": metric_lines_cn,
            },
            "run_rows": run_rows,
            "strict_conclusion": {
                "core_answer": "The clean execution path now has a runner-level status view. It can tell which batch outputs already exist and which launch commands still need to be executed.",
                "main_hard_gaps": [
                    "the runner itself does not execute the model; it only tracks launchable and completed states",
                    "real progress now depends on whether the local DeepSeek model can actually run inside the current environment",
                ],
            },
        }
