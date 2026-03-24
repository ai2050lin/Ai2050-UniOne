#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import torch
from transformers import AutoTokenizer

from stage119_gpt2_embedding_full_vocab_scan import (
    MODEL_PATH,
    collect_clean_variants,
    load_embedding_weight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage251_delta_position_role_map_20260324"
STAGE248_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage248_delta_position_selection_map_20260324" / "summary.json"

FRUIT_WORDS = {"apple", "pear", "banana", "peach", "orange", "grape", "lemon", "melon"}
ANIMAL_WORDS = {"cat", "dog", "lion", "tiger", "horse", "rabbit", "wolf", "bear"}
TOOL_WORDS = {"knife", "fork", "spoon", "hammer", "saw", "plate", "bottle", "brush"}
BRAND_WORDS = ["iphone", "macbook", "ipad", "apple", "store", "device", "laptop"]
TOP_K = 24


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def choose_token_id(variants_map: Dict[str, list], word: str) -> int:
    variants = variants_map.get(word)
    if not variants:
        raise KeyError(f"未找到词条变体: {word}")
    best = sorted(variants, key=lambda item: (0 if item.leading_space else 1, item.token_id))[0]
    return int(best.token_id)


def load_embeddings_for(words: Iterable[str]) -> Dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        use_fast=False,
    )
    variants_map, _ = collect_clean_variants(tokenizer)
    weight = load_embedding_weight()
    result: Dict[str, torch.Tensor] = {}
    for word in words:
        token_id = choose_token_id(variants_map, word)
        result[word] = weight[token_id].clone()
    return result


def build_brand_delta_dims() -> Set[int]:
    embeddings = load_embeddings_for(BRAND_WORDS)
    brand_stack = torch.stack([embeddings[word] for word in BRAND_WORDS])
    centroid = brand_stack.mean(dim=0)
    dim_counter: Counter[int] = Counter()
    for word, embedding in embeddings.items():
        delta = (embedding - centroid).abs()
        top_idx = torch.topk(delta, k=TOP_K).indices.tolist()
        dim_counter.update(int(idx) for idx in top_idx)
    return {idx for idx, count in dim_counter.items() if count >= 2}


def assign_role(*, dim_index: int, fruit_hits: int, animal_hits: int, tool_hits: int, brand_like: bool, family_hit_count: int) -> str:
    if brand_like:
        return "品牌与跨类触发"
    if family_hit_count >= 3:
        return "跨类共享触发"
    if fruit_hits > animal_hits and fruit_hits > tool_hits:
        return "水果内部差分"
    if animal_hits > fruit_hits and animal_hits > tool_hits:
        return "动物内部差分"
    if tool_hits > fruit_hits and tool_hits > animal_hits:
        return "工具与器物差分"
    if family_hit_count == 1:
        return "单家族局部差分"
    return "混合差分"


def build_summary() -> dict:
    stage248 = load_json(STAGE248_SUMMARY)
    brand_dims = build_brand_delta_dims()
    dim_usage_by_word: Dict[int, Set[str]] = defaultdict(set)
    dim_usage_by_family: Dict[int, Counter[str]] = defaultdict(Counter)

    for row in stage248["word_rows"]:
        family_name = str(row["family_name"])
        word = str(row["word"])
        for dim_index in row["top_delta_dims"]:
            dim_usage_by_word[int(dim_index)].add(word)
            dim_usage_by_family[int(dim_index)][family_name] += 1

    role_rows: List[dict] = []
    role_counter: Counter[str] = Counter()
    top_rows = sorted(stage248["position_rows"], key=lambda row: (-int(row["usage_count"]), -float(row["mean_delta_load"])))
    for row in top_rows:
        dim_index = int(row["dim_index"])
        words = dim_usage_by_word.get(dim_index, set())
        fruit_hits = sum(1 for word in words if word in FRUIT_WORDS)
        animal_hits = sum(1 for word in words if word in ANIMAL_WORDS)
        tool_hits = sum(1 for word in words if word in TOOL_WORDS)
        role_name = assign_role(
            dim_index=dim_index,
            fruit_hits=fruit_hits,
            animal_hits=animal_hits,
            tool_hits=tool_hits,
            brand_like=dim_index in brand_dims,
            family_hit_count=int(row["family_hit_count"]),
        )
        role_counter.update([role_name])
        role_rows.append(
            {
                "dim_index": dim_index,
                "role_name": role_name,
                "usage_count": int(row["usage_count"]),
                "family_hit_count": int(row["family_hit_count"]),
                "fruit_word_hits": fruit_hits,
                "animal_word_hits": animal_hits,
                "tool_word_hits": tool_hits,
                "brand_like": dim_index in brand_dims,
                "base_load": float(row["base_load"]),
                "mean_delta_load": float(row["mean_delta_load"]),
            }
        )

    strongest_role = max(role_counter.items(), key=lambda item: item[1])[0]
    weakest_role = min(role_counter.items(), key=lambda item: item[1])[0]
    role_score = (
        role_counter["水果内部差分"]
        + role_counter["动物内部差分"]
        + role_counter["品牌与跨类触发"]
        + role_counter["跨类共享触发"]
    ) / max(len(role_rows), 1)

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage251_delta_position_role_map",
        "title": "差分位置角色图",
        "status_short": "delta_position_role_map_ready",
        "dimension_count": len(role_rows),
        "brand_dim_count": len(brand_dims),
        "role_score": float(role_score),
        "strongest_role_name": strongest_role,
        "weakest_role_name": weakest_role,
        "top_gap_name": "高频差分维度会集中承担少量固定角色，而不是随机分布在所有位置上",
        "role_counter": dict(role_counter),
        "role_rows": role_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "role_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["role_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["role_rows"])
    report = [
        "# Stage251：差分位置角色图",
        "",
        "## 核心结果",
        f"- 高频差分维度数量：{summary['dimension_count']}",
        f"- 品牌触发相关维度数量：{summary['brand_dim_count']}",
        f"- 角色图总分：{summary['role_score']:.4f}",
        f"- 最强角色：{summary['strongest_role_name']}",
        f"- 最弱角色：{summary['weakest_role_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE251_DELTA_POSITION_ROLE_MAP_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="差分位置角色图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
