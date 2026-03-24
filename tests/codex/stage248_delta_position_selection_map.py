#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from stage119_gpt2_embedding_full_vocab_scan import (
    MODEL_PATH,
    collect_clean_variants,
    load_embedding_weight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage248_delta_position_selection_map_20260324"
TOP_K = 16

FAMILY_WORDS = {
    "fruit": ["apple", "pear", "banana", "peach", "orange", "grape", "lemon", "melon"],
    "animal": ["cat", "dog", "lion", "tiger", "horse", "rabbit", "wolf", "bear"],
    "tool": ["knife", "fork", "spoon", "hammer", "saw", "plate", "bottle", "brush"],
}


def choose_token_id(variants_map: Dict[str, list], word: str) -> int:
    variants = variants_map.get(word)
    if not variants:
        raise KeyError(f"未找到词条变体: {word}")
    best = sorted(variants, key=lambda item: (0 if item.leading_space else 1, item.token_id))[0]
    return int(best.token_id)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        use_fast=False,
    )


def load_family_embeddings() -> Dict[str, Dict[str, torch.Tensor]]:
    tokenizer = load_tokenizer()
    variants_map, _ = collect_clean_variants(tokenizer)
    weight = load_embedding_weight()
    family_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
    for family_name, words in FAMILY_WORDS.items():
        word_map: Dict[str, torch.Tensor] = {}
        for word in words:
            token_id = choose_token_id(variants_map, word)
            word_map[word] = weight[token_id].clone()
        family_embeddings[family_name] = word_map
    return family_embeddings


def build_summary() -> dict:
    family_embeddings = load_family_embeddings()
    dim_count = int(next(iter(next(iter(family_embeddings.values())).values())).shape[0])
    usage_count = torch.zeros(dim_count, dtype=torch.float32)
    family_count = torch.zeros(dim_count, dtype=torch.float32)
    delta_load = torch.zeros(dim_count, dtype=torch.float32)
    base_load = torch.zeros(dim_count, dtype=torch.float32)
    family_hits: Dict[int, set[str]] = {idx: set() for idx in range(dim_count)}
    word_rows: List[dict] = []

    all_embeddings = []
    for family_name, word_map in family_embeddings.items():
        family_stack = torch.stack(list(word_map.values()))
        family_centroid = family_stack.mean(dim=0)
        for word, embedding in word_map.items():
            all_embeddings.append(embedding)
            delta = embedding - family_centroid
            top_vals, top_idx = torch.topk(delta.abs(), k=TOP_K)
            for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
                usage_count[idx] += 1.0
                delta_load[idx] += float(val)
                family_hits[idx].add(family_name)
            top_dims = [int(idx) for idx in top_idx.tolist()]
            word_rows.append(
                {
                    "family_name": family_name,
                    "word": word,
                    "top_delta_dims": top_dims,
                    "mean_top_delta": float(top_vals.mean().item()),
                }
            )

    all_stack = torch.stack(all_embeddings)
    base_load = all_stack.abs().mean(dim=0)
    for idx, families in family_hits.items():
        family_count[idx] = float(len(families))

    global_base_mean = float(base_load.mean().item())
    global_delta_mean = float((delta_load / torch.clamp(usage_count, min=1.0)).mean().item())
    top_dim_idx = torch.topk(usage_count, k=24).indices.tolist()
    top_dim_base_mean = float(base_load[top_dim_idx].mean().item())
    top_dim_delta_mean = float((delta_load[top_dim_idx] / torch.clamp(usage_count[top_dim_idx], min=1.0)).mean().item())
    top_dim_family_spread = float(family_count[top_dim_idx].mean().item()) / len(FAMILY_WORDS)
    low_base_high_delta_ratio = sum(
        1
        for idx in top_dim_idx
        if float(base_load[idx]) < global_base_mean and float(delta_load[idx] / max(usage_count[idx].item(), 1.0)) > global_delta_mean
    ) / len(top_dim_idx)
    position_score = (low_base_high_delta_ratio + top_dim_family_spread + min(1.0, top_dim_delta_mean / (global_delta_mean + 1e-9))) / 3.0

    position_rows = []
    for idx in top_dim_idx:
        mean_delta = float(delta_load[idx] / max(usage_count[idx].item(), 1.0))
        position_rows.append(
            {
                "dim_index": int(idx),
                "usage_count": int(usage_count[idx].item()),
                "family_hit_count": int(family_count[idx].item()),
                "base_load": float(base_load[idx].item()),
                "mean_delta_load": mean_delta,
                "is_low_base_high_delta": bool(float(base_load[idx]) < global_base_mean and mean_delta > global_delta_mean),
            }
        )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage248_delta_position_selection_map",
        "title": "差分位置选择图",
        "status_short": "delta_position_selection_map_ready",
        "family_count": len(FAMILY_WORDS),
        "word_count": len(word_rows),
        "top_k": TOP_K,
        "global_base_mean": global_base_mean,
        "global_delta_mean": global_delta_mean,
        "top_dim_base_mean": top_dim_base_mean,
        "top_dim_delta_mean": top_dim_delta_mean,
        "top_dim_family_spread": top_dim_family_spread,
        "low_base_high_delta_ratio": low_base_high_delta_ratio,
        "position_score": position_score,
        "strongest_gap_name": "差分位置集中在低基底负载且高局部增量的少量维度",
        "position_rows": position_rows,
        "word_rows": word_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "position_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["position_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["position_rows"])
    with (output_dir / "word_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["word_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["word_rows"])
    report = [
        "# Stage248：差分位置选择图",
        "",
        "## 核心结果",
        f"- 家族数量：{summary['family_count']}",
        f"- 词数量：{summary['word_count']}",
        f"- 差分前 {summary['top_k']} 维重复使用位置图已生成",
        f"- 全局基底均值：{summary['global_base_mean']:.4f}",
        f"- 全局差分均值：{summary['global_delta_mean']:.4f}",
        f"- 高频差分位置基底均值：{summary['top_dim_base_mean']:.4f}",
        f"- 高频差分位置差分均值：{summary['top_dim_delta_mean']:.4f}",
        f"- 低基底高差分比例：{summary['low_base_high_delta_ratio']:.4f}",
        f"- 位置图总分：{summary['position_score']:.4f}",
        f"- 头号发现：{summary['strongest_gap_name']}",
    ]
    (output_dir / "STAGE248_DELTA_POSITION_SELECTION_MAP_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="差分位置选择图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
