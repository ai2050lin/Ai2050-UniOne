from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            if len(row) >= 2:
                noun = str(row[0]).strip()
                category = str(row[1]).strip()
                if noun and category:
                    rows.append((noun, category))
    return rows


@dataclass
class DnnThousandNounSourceGapBoard:
    root: Path
    thousand_program: Dict[str, Any]

    @classmethod
    def from_repo(cls, root: Path) -> "DnnThousandNounSourceGapBoard":
        temp = root / "tests" / "codex_temp"
        return cls(
            root=root,
            thousand_program=load_json(temp / "dnn_thousand_noun_family_patch_offset_program_block_20260316.json"),
        )

    def summary(self) -> Dict[str, Any]:
        base_csv = self.root / "tests" / "codex" / "deepseek7b_bilingual_nouns.csv"
        utf8_csv = self.root / "tests" / "codex" / "deepseek7b_bilingual_nouns_utf8.csv"
        base_rows = load_csv_rows(base_csv)
        utf8_rows = load_csv_rows(utf8_csv)

        base_unique = len({noun for noun, _ in base_rows})
        utf8_unique = len({noun for noun, _ in utf8_rows})
        base_cat = Counter(category for _, category in base_rows)
        utf8_cat = Counter(category for _, category in utf8_rows)

        unique_target = 3000
        unique_gap = max(0, unique_target - base_unique)
        prompt_multiplier_not_enough = unique_target > base_unique

        category_rows = [
            {
                "category": category,
                "base_count": int(base_cat[category]),
                "seed_count": int(utf8_cat.get(category, 0)),
            }
            for category in sorted(base_cat.keys())
        ]

        metric_lines_cn = [
            f"（当前基础唯一名词数）base_unique_nouns = {base_unique:.4f}",
            f"（当前种子唯一名词数）seed_unique_nouns = {utf8_unique:.4f}",
            f"（3000唯一名词缺口）unique_noun_gap_to_3000 = {unique_gap:.4f}",
            f"（当前类别数量）category_count = {len(base_cat):.4f}",
            f"（最大单类名词数）max_category_size = {max(base_cat.values()):.4f}",
            f"（多提示词不能替代唯一名词）prompt_multiplier_not_enough = {1.0 if prompt_multiplier_not_enough else 0.0:.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "source_state": {
                "base_csv": str(base_csv.relative_to(self.root)),
                "utf8_seed_csv": str(utf8_csv.relative_to(self.root)),
                "base_unique_nouns": base_unique,
                "seed_unique_nouns": utf8_unique,
                "category_rows": category_rows,
            },
            "gap_analysis": {
                "unique_target": unique_target,
                "unique_gap": unique_gap,
                "prompt_multiplier_not_enough": prompt_multiplier_not_enough,
                "why": [
                    "更多提示词只会增加同一个名词的上下文覆盖，不会增加唯一名词种类。",
                    "要做真正的数千名词 family patch / concept offset 分析，必须先扩真实唯一名词词表。",
                    "当前 280 个唯一名词足以做 hundreds-scale 主线，但还不够诚实地叫 thousands-scale unique noun atlas。",
                ],
            },
            "execution_blocks": [
                {
                    "block": "阶段一：把 280 唯一名词做成稳定 hundreds-scale atlas",
                    "content": "先把现有基础词表完整接到 mass noun 和 specific dense export 管线里，形成稳定基线。",
                },
                {
                    "block": "阶段二：扩真实唯一名词源到 1000+",
                    "content": "增加同家族新名词、同义词、跨领域名词，而不是只增加 prompt 模板。",
                },
                {
                    "block": "阶段三：扩到 3000 唯一名词并重算编码规律",
                    "content": "在足够大的唯一词表上重算 family patch 稳定性、offset 稀疏性和共享 scaffold。",
                },
            ],
            "strict_conclusion": {
                "core_answer": "Yes, thousands-scale noun analysis is still a good route. But the current honest base is 280 unique nouns, not 3000. So the immediate task is not pretending prompt expansion equals noun scaling; it is expanding the real unique noun source itself.",
                "main_hard_gaps": [
                    "current unique noun lexicon is still only hundreds-scale",
                    "prompt multiplication cannot replace true unique noun scaling",
                    "without more real nouns, thousands-scale claims would be overstated",
                ],
            },
        }
