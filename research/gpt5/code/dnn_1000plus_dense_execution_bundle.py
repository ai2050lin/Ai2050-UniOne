from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple
import csv
import json

from research.gpt5.code.dnn_1000plus_noun_source_builder import Dnn1000PlusNounSourceBuilder


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class NounRow:
    noun: str
    category: str


@dataclass
class Dnn1000PlusDenseExecutionBundle:
    root: Path
    noun_rows: List[NounRow]
    dense_schema: Dict[str, Any]
    specific_manifest: Dict[str, Any]

    @classmethod
    def from_repo(cls, root: Path) -> "Dnn1000PlusDenseExecutionBundle":
        noun_csv = root / "tests" / "codex" / "deepseek7b_bilingual_nouns_1000plus.csv"
        if not noun_csv.exists():
            Dnn1000PlusNounSourceBuilder(root).write_csv("tests/codex/deepseek7b_bilingual_nouns_1000plus.csv")

        noun_rows: List[NounRow] = []
        with noun_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#") or len(row) < 2:
                    continue
                noun_rows.append(NounRow(noun=row[0].strip(), category=row[1].strip()))

        temp = root / "tests" / "codex_temp"
        dense_schema = load_json(temp / "dnn_joint_dense_export_schema_block_20260315.json")
        specific_manifest = load_json(temp / "dnn_family_specific_dense_target_manifest_block_20260316.json")
        return cls(
            root=root,
            noun_rows=noun_rows,
            dense_schema=dense_schema,
            specific_manifest=specific_manifest,
        )

    def _target_groups(self) -> List[Dict[str, Any]]:
        return self.specific_manifest["target_groups"]

    def _seed_batches(self) -> Tuple[List[List[NounRow]], Dict[str, bool]]:
        noun_map: Dict[str, NounRow] = {}
        for row in self.noun_rows:
            noun_map.setdefault(row.noun, row)

        seeded: List[List[NounRow]] = []
        used: Dict[str, bool] = {}
        for group in self._target_groups():
            batch_rows: List[NounRow] = []
            for concept in group["concepts"]:
                if concept in noun_map:
                    batch_rows.append(noun_map[concept])
            if batch_rows:
                seeded.append(batch_rows)
                for row in batch_rows:
                    used[row.noun] = True
        return seeded, used

    def _balanced_fill(self, seeded: List[List[NounRow]], used: Dict[str, bool], batch_size: int) -> List[List[NounRow]]:
        category_pool: Dict[str, Deque[NounRow]] = defaultdict(deque)
        for row in self.noun_rows:
            if used.get(row.noun):
                continue
            category_pool[row.category].append(row)

        categories = sorted(category_pool.keys())
        batches = [list(batch) for batch in seeded]

        # Fill seeded batches first so that control groups remain embedded but still become balanced.
        for batch in batches:
            while len(batch) < batch_size:
                progress = False
                present = {row.category for row in batch}
                for category in categories:
                    if len(batch) >= batch_size:
                        break
                    if category_pool[category] and category not in present:
                        row = category_pool[category].popleft()
                        batch.append(row)
                        used[row.noun] = True
                        progress = True
                        present.add(category)
                if len(batch) >= batch_size:
                    break
                for category in categories:
                    if len(batch) >= batch_size:
                        break
                    if category_pool[category]:
                        row = category_pool[category].popleft()
                        batch.append(row)
                        used[row.noun] = True
                        progress = True
                if not progress:
                    break

        current: List[NounRow] = []
        while True:
            progress = False
            for category in categories:
                if category_pool[category]:
                    row = category_pool[category].popleft()
                    current.append(row)
                    used[row.noun] = True
                    progress = True
                    if len(current) >= batch_size:
                        batches.append(current)
                        current = []
            if not progress:
                break
        if current:
            batches.append(current)
        return batches

    def _write_batch_csvs(self, batches: List[List[NounRow]], out_dir: Path) -> List[Dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        task_rows: List[Dict[str, Any]] = []
        group_map = {group["name"]: set(group["concepts"]) for group in self._target_groups()}
        for idx, batch in enumerate(batches, start=1):
            rel_csv = Path("tests") / "codex_temp" / "dnn_1000plus_batches" / f"batch_{idx:03d}.csv"
            csv_path = self.root / rel_csv
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["# noun", "category"])
                for row in batch:
                    writer.writerow([row.noun, row.category])

            categories = sorted({row.category for row in batch})
            nouns = {row.noun for row in batch}
            matched_groups = [name for name, concepts in group_map.items() if concepts & nouns]
            mass_out_dir = f"tempdata/dnn_1000plus_mass_noun_scan_batch_{idx:03d}"
            command = (
                "python tests/codex/deepseek7b_mass_noun_encoding_scan.py "
                f"--nouns-file \"{rel_csv.as_posix()}\" "
                f"--max-nouns {len(batch)} "
                "--local-files-only "
                f"--output-dir \"{mass_out_dir}\""
            )
            task_rows.append(
                {
                    "batch_id": idx,
                    "csv_relative_path": rel_csv.as_posix(),
                    "noun_count": len(batch),
                    "category_count": len(categories),
                    "categories": categories,
                    "anchor_target_groups": matched_groups,
                    "launch_command": command,
                    "mass_output_dir": mass_out_dir,
                }
            )
        return task_rows

    def summary(self, batch_size: int = 120) -> Dict[str, Any]:
        seeded, used = self._seed_batches()
        batches = self._balanced_fill(seeded, used, batch_size=batch_size)
        task_rows = self._write_batch_csvs(
            batches=batches,
            out_dir=self.root / "tests" / "codex_temp" / "dnn_1000plus_batches",
        )

        total_nouns = sum(row["noun_count"] for row in task_rows)
        avg_category_coverage = sum(row["category_count"] for row in task_rows) / max(1, len(task_rows))
        full_category_batches = sum(1 for row in task_rows if row["category_count"] >= 10)
        anchored_batches = sum(1 for row in task_rows if row["anchor_target_groups"])
        dense_schema_rows = self.dense_schema["headline_metrics"]["schema_rows"]
        specific_schema = next(row for row in dense_schema_rows if row["bucket_name"] == "specific_dense_signature")

        stage_ready = min(
            1.0,
            0.35 * min(1.0, total_nouns / 1000.0)
            + 0.25 * min(1.0, avg_category_coverage / 10.0)
            + 0.20 * min(1.0, anchored_batches / 3.0)
            + 0.20 * (1.0 if specific_schema["launchable"] else 0.0),
        )
        projected_family_patch_stability = min(0.95, 0.7846 + 0.08 * min(1.0, avg_category_coverage / 10.0) + 0.04 * min(1.0, anchored_batches / 3.0))
        projected_concept_offset_sparsity = min(0.98, 0.9631 + 0.01 * min(1.0, total_nouns / 1000.0))

        metric_lines_cn = [
            f"（1000+分批总数）batch_count = {len(task_rows):.4f}",
            f"（1000+总名词数）total_batched_nouns = {total_nouns:.4f}",
            f"（平均类别覆盖数）avg_category_coverage = {avg_category_coverage:.4f}",
            f"（全类别覆盖批次数）full_category_batch_count = {full_category_batches:.4f}",
            f"（specific锚点批次数）anchored_batch_count = {anchored_batches:.4f}",
            f"（specific导出schema可启动）specific_schema_launchable = {1.0 if specific_schema['launchable'] else 0.0:.4f}",
            f"（1000+执行阶段准备度）execution_stage_readiness = {stage_ready:.4f}",
            f"（投影family稳定性）projected_family_patch_stability = {projected_family_patch_stability:.4f}",
            f"（投影offset稀疏恢复）projected_concept_offset_sparsity = {projected_concept_offset_sparsity:.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "task_rows": task_rows,
            "strict_conclusion": {
                "core_answer": "The 1000+ noun source is no longer just a CSV. It is now partitioned into launchable balanced batches, wired to the existing mass noun scan entrypoint, and anchored to specific-dense target groups for family-to-specific closure work.",
                "main_hard_gaps": [
                    "the bundle is launchable but the heavy model inference has not been executed yet",
                    "successor exact closure is still not improved by this noun-scaling bundle alone",
                    "the current outputs are execution artifacts and projected stage targets, not post-harvest theorem closure",
                ],
            },
            "headline_metrics": {
                "batch_count": len(task_rows),
                "total_batched_nouns": total_nouns,
                "avg_category_coverage": avg_category_coverage,
                "full_category_batch_count": full_category_batches,
                "anchored_batch_count": anchored_batches,
                "specific_schema_launchable": 1.0 if specific_schema["launchable"] else 0.0,
                "execution_stage_readiness": stage_ready,
                "projected_family_patch_stability": projected_family_patch_stability,
                "projected_concept_offset_sparsity": projected_concept_offset_sparsity,
                "metric_lines_cn": metric_lines_cn,
            },
        }
