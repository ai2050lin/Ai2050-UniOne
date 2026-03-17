from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json

from research.gpt5.code.dnn_hundreds_scale_noun_atlas_baseline import DnnHundredsScaleNounAtlasBaseline
from research.gpt5.code.dnn_1000plus_noun_source_builder import Dnn1000PlusNounSourceBuilder


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Dnn3000NounRolloutProgram:
    baseline: Dict[str, Any]
    source_1000: Dict[str, Any]
    source_gap: Dict[str, Any]

    @classmethod
    def from_repo(cls, root: Path) -> "Dnn3000NounRolloutProgram":
        temp = root / "tests" / "codex_temp"
        baseline_path = temp / "dnn_hundreds_scale_noun_atlas_baseline_block_20260316.json"
        source_1000_path = temp / "dnn_1000plus_noun_source_builder_block_20260316.json"
        source_gap_path = temp / "dnn_thousand_noun_source_gap_board_block_20260316.json"

        if baseline_path.exists():
            baseline = load_json(baseline_path)
        else:
            baseline = {
                "baseline": DnnHundredsScaleNounAtlasBaseline.from_repo(root).summary()["baseline"]
            }

        if source_1000_path.exists():
            source_1000 = load_json(source_1000_path)
        else:
            builder = Dnn1000PlusNounSourceBuilder(root)
            out_csv = builder.write_csv("tests/codex/deepseek7b_bilingual_nouns_1000plus.csv")
            import csv
            from collections import Counter

            rows = []
            with out_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].startswith("#") or len(row) < 2:
                        continue
                    rows.append((row[0], row[1]))
            category_counter = Counter(category for _, category in rows)
            source_1000 = {
                "category_rows": [
                    {"category": category, "count": int(count)}
                    for category, count in sorted(category_counter.items())
                ]
            }

        source_gap = load_json(source_gap_path)
        return cls(
            baseline=baseline,
            source_1000=source_1000,
            source_gap=source_gap,
        )

    def summary(self) -> Dict[str, Any]:
        hundreds = int(self.baseline["baseline"]["hundreds_unique_nouns"])
        source_1000 = sum(row["count"] for row in self.source_1000["category_rows"])
        gap_to_3000 = max(0, 3000 - source_1000)

        phase1_target = hundreds
        phase2_target = source_1000
        phase3_target = 3000

        rollout_ready_score = min(
            1.0,
            0.30 * min(1.0, phase1_target / 280.0)
            + 0.35 * min(1.0, phase2_target / 1000.0)
            + 0.35 * min(1.0, (phase2_target + 500) / 1500.0),
        )

        metric_lines_cn = [
            f"（阶段一目标名词数）phase1_target_nouns = {phase1_target:.4f}",
            f"（阶段二目标名词数）phase2_target_nouns = {phase2_target:.4f}",
            f"（阶段三目标名词数）phase3_target_nouns = {phase3_target:.4f}",
            f"（1000+到3000缺口）gap_from_1000plus_to_3000 = {gap_to_3000:.4f}",
            f"（3000计划阶段数量）rollout_phase_count = {3.0:.4f}",
            f"（3000计划准备度）rollout_ready_score = {rollout_ready_score:.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "phases": [
                {
                    "phase": "阶段一：hundreds-scale 稳定基线",
                    "target_nouns": phase1_target,
                    "content": "把现有 280 唯一名词完整接入 mass noun scan、family atlas、specific dense export，形成稳定 baseline。",
                },
                {
                    "phase": "阶段二：1000+ 唯一名词扩张",
                    "target_nouns": phase2_target,
                    "content": "把 1000+ 词源接入同一管线，不再只跑小词表局部例子，开始重算全局 family patch 与 concept offset 统计。",
                },
                {
                    "phase": "阶段三：3000 唯一名词定理冲刺",
                    "target_nouns": phase3_target,
                    "content": "继续扩真实唯一名词到 3000，并在该尺度上重算 family patch 稳定性、offset 稀疏性、共享 scaffold 和具体概念恢复边界。",
                },
            ],
            "gap_from_1000plus_to_3000": gap_to_3000,
            "strict_conclusion": {
                "core_answer": "The repo now has all three pieces of the noun-scaling route: a real hundreds-scale baseline, a generated 1000+ noun source, and a concrete 3000-noun rollout program. The remaining issue is no longer missing structure, but executing the rollout through the dense analysis pipeline.",
                "main_hard_gaps": [
                    "the 3000-noun stage still needs more real unique nouns beyond the generated 1000+ source",
                    "running larger noun sets through dense harvesting will be the expensive step",
                    "even a 3000-noun rollout will still need successor exact closure work in parallel",
                ],
            },
        }
