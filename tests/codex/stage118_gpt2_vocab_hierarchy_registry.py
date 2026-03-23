#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage118: GPT-2 词表层级注册块

目标：
1. 从 GPT-2 全词表里分离出可用的干净概念词。
2. 按 Micro / Meso / Macro（微观 / 中观 / 宏观）做首轮注册。
3. 为后续“家族共享基底 + 实例偏置”分析提供可追踪的词表底座。
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    r"d:\develop\model\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage118_gpt2_vocab_hierarchy_registry_20260323"
ENGLISH_NOUNS_CSV = PROJECT_ROOT / "tests" / "codex" / "deepseek7b_nouns_english_520_clean.csv"


MICRO_SEEDS = {
    "red",
    "green",
    "blue",
    "yellow",
    "black",
    "white",
    "sweet",
    "sour",
    "bitter",
    "salty",
    "heavy",
    "light",
    "soft",
    "hard",
    "round",
    "sharp",
    "smooth",
    "rough",
    "warm",
    "cold",
    "hot",
    "small",
    "large",
    "big",
    "tiny",
    "thick",
    "thin",
    "bright",
    "dark",
    "ripe",
    "fresh",
    "juicy",
    "sweetest",
    "edible",
    "fragrant",
    "beautiful",
    "ugly",
    "quick",
    "slow",
    "heavy",
    "dense",
    "solid",
    "liquid",
    "wooden",
    "metallic",
    "glossy",
    "sticky",
    "dry",
    "wet",
}

MACRO_VERB_SEEDS = {
    "run",
    "walk",
    "jump",
    "eat",
    "drink",
    "think",
    "learn",
    "speak",
    "write",
    "read",
    "move",
    "change",
    "grow",
    "build",
    "destroy",
    "create",
    "reason",
    "argue",
    "prove",
    "judge",
    "choose",
    "remember",
    "forget",
    "explain",
    "compare",
    "classify",
    "imagine",
    "reflect",
    "travel",
    "transport",
    "connect",
    "divide",
    "unify",
    "expand",
    "compress",
    "govern",
    "organize",
    "stabilize",
    "predict",
}

MACRO_ABSTRACT_SEEDS = {
    "justice",
    "truth",
    "freedom",
    "beauty",
    "love",
    "hope",
    "fear",
    "wisdom",
    "logic",
    "language",
    "memory",
    "infinity",
    "system",
    "structure",
    "meaning",
    "identity",
    "order",
    "chaos",
    "history",
    "future",
    "theory",
    "principle",
    "symmetry",
    "topology",
    "geometry",
    "mathematics",
    "ethics",
    "science",
    "cause",
    "effect",
}

MICRO_SUFFIXES = ("ous", "ful", "less", "ive", "able", "ible", "ish", "ary", "al", "ic", "y")
MACRO_SUFFIXES = (
    "ness",
    "tion",
    "sion",
    "ity",
    "ism",
    "hood",
    "ship",
    "ment",
    "ance",
    "ence",
    "dom",
    "acy",
    "ure",
)
VERB_SUFFIXES = ("ing", "ed", "ize", "ise", "fy", "ate", "en")

WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]{1,29}$")


@dataclass
class TokenVariant:
    token_id: int
    raw_token: str
    normalized: str


def discover_tokenizer() -> AutoTokenizer:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到本地 GPT-2 模型路径: {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, use_fast=False)
    return tok


def load_meso_seed_map() -> Dict[str, str]:
    seed_map: Dict[str, str] = {}
    with ENGLISH_NOUNS_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.reader(row for row in fh if not row.startswith("#"))
        for row in reader:
            if len(row) < 2:
                continue
            noun = row[0].strip().lower()
            category = row[1].strip().lower()
            if category == "abstract":
                continue
            seed_map[noun] = category
    return seed_map


def normalize_gpt2_token(raw_token: str) -> str:
    cleaned = raw_token.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
    return cleaned.strip().lower()


def display_token(raw_token: str) -> str:
    return raw_token.replace("Ġ", "<SP>").replace("Ċ", "<NL>").replace("ĉ", "<TAB>")


def is_clean_lexical_word(word: str) -> bool:
    if not word:
        return False
    if not WORD_RE.fullmatch(word):
        return False
    if word.count("'") > 1 or word.count("-") > 1:
        return False
    if len(word) < 2 or len(word) > 30:
        return False
    return True


def collect_clean_variants(tokenizer: AutoTokenizer) -> Tuple[Dict[str, List[TokenVariant]], int]:
    variants: Dict[str, List[TokenVariant]] = defaultdict(list)
    fragment_count = 0
    for token_id in range(int(tokenizer.vocab_size)):
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        normalized = normalize_gpt2_token(raw_token)
        if not is_clean_lexical_word(normalized):
            fragment_count += 1
            continue
        variants[normalized].append(TokenVariant(token_id=token_id, raw_token=raw_token, normalized=normalized))
    return variants, fragment_count


def score_word(word: str, meso_seed_map: Dict[str, str]) -> Tuple[str, Dict[str, float], str]:
    scores = {"micro": 0.0, "meso": 0.0, "macro": 0.0}
    reasons: List[str] = []

    if word in MICRO_SEEDS:
        scores["micro"] += 3.0
        reasons.append("micro_seed")
    if word in meso_seed_map:
        scores["meso"] += 4.0
        reasons.append(f"meso_seed:{meso_seed_map[word]}")
    if word in MACRO_VERB_SEEDS:
        scores["macro"] += 3.0
        reasons.append("macro_verb_seed")
    if word in MACRO_ABSTRACT_SEEDS:
        scores["macro"] += 4.0
        reasons.append("macro_abstract_seed")

    if word.endswith(MICRO_SUFFIXES):
        scores["micro"] += 1.2
        reasons.append("micro_suffix")
    if word.endswith(MACRO_SUFFIXES):
        scores["macro"] += 1.8
        reasons.append("macro_suffix")
    if word.endswith(VERB_SUFFIXES):
        scores["macro"] += 1.0
        reasons.append("verb_suffix")

    if word in {"fruit", "animal", "vehicle", "object", "teacher", "doctor", "king", "queen"}:
        scores["meso"] += 2.0
        reasons.append("meso_anchor")
    if word in {"color", "shape", "size", "taste", "texture", "weight"}:
        scores["micro"] += 1.8
        reasons.append("micro_anchor")
    if word in {"truth", "logic", "justice", "theory", "system", "infinity"}:
        scores["macro"] += 2.0
        reasons.append("macro_anchor")

    top_label, top_score = max(scores.items(), key=lambda item: item[1])
    runner_up = sorted(scores.values(), reverse=True)[1]
    margin = top_score - runner_up
    if top_score < 2.0 or margin < 0.6:
        return "unknown", scores, ",".join(reasons) if reasons else "weak_signal"
    return top_label, scores, ",".join(reasons)


def build_registry(tokenizer: AutoTokenizer) -> Tuple[Dict[str, object], Dict[str, object]]:
    meso_seed_map = load_meso_seed_map()
    variants, fragment_count = collect_clean_variants(tokenizer)

    registry_rows: List[Dict[str, object]] = []
    counts = Counter()
    category_hist = Counter()
    confidence_margins: List[float] = []

    for word, word_variants in sorted(variants.items()):
        label, scores, reason = score_word(word, meso_seed_map)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = float(sorted_scores[0] - sorted_scores[1])
        if label != "unknown":
            confidence_margins.append(margin)
        counts[label] += 1
        if label == "meso":
            category_hist[meso_seed_map.get(word, "generic")] += 1
        registry_rows.append(
            {
                "word": word,
                "label": label,
                "scores": scores,
                "reason": reason,
                "variant_count": len(word_variants),
                "token_ids": [item.token_id for item in word_variants[:8]],
                "raw_tokens": [display_token(item.raw_token) for item in word_variants[:8]],
            }
        )

    clean_count = len(registry_rows)
    registered_count = counts["micro"] + counts["meso"] + counts["macro"]
    confidence_mean = float(sum(confidence_margins) / max(1, len(confidence_margins)))
    family_coverage = float(
        min(
            1.0,
            sum(1 for _, n in category_hist.items() if n >= 4) / max(1, len(set(meso_seed_map.values()))),
        )
    )
    clean_ratio = clean_count / max(1, tokenizer.vocab_size)
    registered_ratio = registered_count / max(1, clean_count)
    fragment_ratio = fragment_count / max(1, tokenizer.vocab_size)
    basis_offset_registry_score = float(
        0.28 * registered_ratio
        + 0.24 * min(1.0, confidence_mean / 3.0)
        + 0.24 * family_coverage
        + 0.24 * (1.0 - fragment_ratio)
    )

    anchor_words = {}
    for probe in ("apple", "banana", "fruit", "red", "sweet", "justice", "truth", "run", "jump"):
        row = next((item for item in registry_rows if item["word"] == probe), None)
        anchor_words[probe] = row

    summary = {
        "stage": "stage118_gpt2_vocab_hierarchy_registry",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "gpt2",
        "vocab_size": int(tokenizer.vocab_size),
        "fragment_token_count": int(fragment_count),
        "fragment_ratio": float(fragment_ratio),
        "clean_unique_word_count": int(clean_count),
        "clean_unique_word_ratio": float(clean_ratio),
        "registered_unique_word_count": int(registered_count),
        "registered_unique_word_ratio": float(registered_ratio),
        "micro_count": int(counts["micro"]),
        "meso_count": int(counts["meso"]),
        "macro_count": int(counts["macro"]),
        "unknown_clean_count": int(counts["unknown"]),
        "registry_confidence_mean": confidence_mean,
        "meso_family_category_count": int(len(category_hist)),
        "meso_family_coverage": family_coverage,
        "basis_offset_registry_score": basis_offset_registry_score,
        "anchor_words": anchor_words,
        "top_meso_categories": category_hist.most_common(10),
        "status_short": "gpt2_vocab_registry_ready" if basis_offset_registry_score >= 0.55 else "gpt2_vocab_registry_transition",
    }

    registry = {
        "meta": {
            "model_name": "gpt2",
            "timestamp": summary["timestamp"],
            "description": "GPT-2 词表 Micro/Meso/Macro 首轮注册表",
        },
        "rows": registry_rows,
    }
    return summary, registry


def write_outputs(summary: Dict[str, object], registry: Dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    registry_path = OUTPUT_DIR / "registry.json"
    report_path = OUTPUT_DIR / "REPORT.md"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Stage118 GPT-2 词表层级注册报告",
        "",
        f"- 时间: {summary['timestamp']}",
        f"- 词表大小: {summary['vocab_size']}",
        f"- 干净概念词数量: {summary['clean_unique_word_count']}",
        f"- 已注册概念词数量: {summary['registered_unique_word_count']}",
        f"- Micro（微观）数量: {summary['micro_count']}",
        f"- Meso（中观）数量: {summary['meso_count']}",
        f"- Macro（宏观）数量: {summary['macro_count']}",
        f"- Unknown（未确定）数量: {summary['unknown_clean_count']}",
        f"- 家族覆盖率: {summary['meso_family_coverage']:.6f}",
        f"- 注册总分: {summary['basis_offset_registry_score']:.6f}",
        "",
        "## 锚点词检查",
    ]
    for word, row in summary["anchor_words"].items():
        if row is None:
            report_lines.append(f"- {word}: 未出现在可用干净词集合里")
            continue
        report_lines.append(
            f"- {word}: label={row['label']}, scores={row['scores']}, reason={row['reason']}, variants={row['variant_count']}"
        )
    report_lines.extend(
        [
            "",
            "## 当前结论",
            "- 这一步解决的是“GPT-2 全词表中哪些词可以进入理论分析底座”。",
            "- 它不是最终理论，而是为后续家族共享基底、实例偏置、关系偏置提供干净注册表。",
            "- 当前最关键对象是可注册的 Meso（中观）实体词，因为水果族共享基底分析首先依赖它们。",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tokenizer = discover_tokenizer()
    summary, registry = build_registry(tokenizer)
    write_outputs(summary, registry)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
