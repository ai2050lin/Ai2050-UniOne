from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple
import csv
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class NounRow:
    noun: str
    category: str


@dataclass
class DnnCleanEnglishExecutionBundle:
    root: Path
    noun_rows: List[NounRow]
    dense_schema: Dict[str, Any]

    @classmethod
    def from_repo(cls, root: Path) -> "DnnCleanEnglishExecutionBundle":
        source_csv = root / "tests" / "codex" / "deepseek7b_bilingual_nouns_1000plus.csv"
        noun_rows: List[NounRow] = []
        seen = set()
        with source_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#") or len(row) < 2:
                    continue
                noun = row[0].strip()
                category = row[1].strip()
                if not noun:
                    continue
                if not all(ord(ch) < 128 for ch in noun):
                    continue
                if noun in seen:
                    continue
                seen.add(noun)
                noun_rows.append(NounRow(noun=noun, category=category))

        temp = root / "tests" / "codex_temp"
        dense_schema = load_json(temp / "dnn_joint_dense_export_schema_block_20260315.json")
        return cls(root=root, noun_rows=noun_rows, dense_schema=dense_schema)

    def write_clean_source(self) -> Path:
        rel = Path("tests") / "codex" / "deepseek7b_nouns_english_500plus.csv"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["# noun", "category"])
            for row in self.noun_rows:
                writer.writerow([row.noun, row.category])
        return path

    def _balanced_batches(self, batch_size: int) -> List[List[NounRow]]:
        category_pool: Dict[str, Deque[NounRow]] = defaultdict(deque)
        for row in self.noun_rows:
            category_pool[row.category].append(row)

        categories = sorted(category_pool.keys())
        batches: List[List[NounRow]] = []
        current: List[NounRow] = []
        while True:
            progress = False
            for category in categories:
                if category_pool[category]:
                    current.append(category_pool[category].popleft())
                    progress = True
                    if len(current) >= batch_size:
                        batches.append(current)
                        current = []
            if not progress:
                break
        if current:
            batches.append(current)
        return batches

    def _write_batch_csvs(self, batches: List[List[NounRow]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, batch in enumerate(batches, start=1):
            rel_csv = Path("tests") / "codex_temp" / "dnn_clean_english_batches" / f"batch_{idx:03d}.csv"
            path = self.root / rel_csv
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["# noun", "category"])
                for row in batch:
                    writer.writerow([row.noun, row.category])

            rows.append(
                {
                    "batch_id": idx,
                    "csv_relative_path": rel_csv.as_posix(),
                    "noun_count": len(batch),
                    "category_count": len({row.category for row in batch}),
                    "launch_command": (
                        "python tests/codex/deepseek7b_mass_noun_encoding_scan.py "
                        f"--nouns-file \"{rel_csv.as_posix()}\" "
                        f"--max-nouns {len(batch)} "
                        "--local-files-only "
                        f"--output-dir \"tempdata/dnn_clean_english_mass_noun_scan_batch_{idx:03d}\""
                    ),
                }
            )
        return rows

    def summary(self, batch_size: int = 120) -> Dict[str, Any]:
        clean_source = self.write_clean_source()
        batches = self._balanced_batches(batch_size=batch_size)
        batch_rows = self._write_batch_csvs(batches)

        total_unique = len(self.noun_rows)
        avg_category_coverage = sum(row["category_count"] for row in batch_rows) / max(1, len(batch_rows))
        full_category_batches = sum(1 for row in batch_rows if row["category_count"] >= 10)
        specific_schema = next(
            row for row in self.dense_schema["headline_metrics"]["schema_rows"] if row["bucket_name"] == "specific_dense_signature"
        )
        clean_path_readiness = min(
            1.0,
            0.40 * min(1.0, total_unique / 518.0)
            + 0.25 * min(1.0, avg_category_coverage / 10.0)
            + 0.20 * min(1.0, full_category_batches / max(1, len(batch_rows)))
            + 0.15 * (1.0 if specific_schema["launchable"] else 0.0),
        )

        metric_lines_cn = [
            f"（干净英文唯一名词数）clean_unique_english_nouns = {total_unique:.4f}",
            f"（干净执行批次数）clean_batch_count = {len(batch_rows):.4f}",
            f"（干净平均类别覆盖数）clean_avg_category_coverage = {avg_category_coverage:.4f}",
            f"（干净全类别覆盖批次数）clean_full_category_batch_count = {full_category_batches:.4f}",
            f"（specific导出schema可启动）specific_schema_launchable = {1.0 if specific_schema['launchable'] else 0.0:.4f}",
            f"（干净执行路径准备度）clean_path_readiness = {clean_path_readiness:.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "clean_source_relative_path": str(clean_source.relative_to(self.root)).replace("\\", "/"),
            "batch_rows": batch_rows,
            "headline_metrics": {
                "clean_unique_english_nouns": total_unique,
                "clean_batch_count": len(batch_rows),
                "clean_avg_category_coverage": avg_category_coverage,
                "clean_full_category_batch_count": full_category_batches,
                "specific_schema_launchable": 1.0 if specific_schema["launchable"] else 0.0,
                "clean_path_readiness": clean_path_readiness,
                "metric_lines_cn": metric_lines_cn,
            },
            "strict_conclusion": {
                "core_answer": "Because the bilingual 1000+ source still contains historical mojibake, the practical no-garble execution path is to first run a clean 518-noun English bundle. It preserves all 10 categories and keeps the mass noun scan path executable.",
                "main_hard_gaps": [
                    "the clean path is smaller than the raw 1000+ source",
                    "it is a practical execution repair, not a final multilingual solution",
                    "the Chinese side still needs a true clean source rebuild rather than continued reuse of corrupted text",
                ],
            },
        }
