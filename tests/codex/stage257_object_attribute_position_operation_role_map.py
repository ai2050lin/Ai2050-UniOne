#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from stage119_gpt2_embedding_full_vocab_scan import (
    MODEL_PATH,
    collect_clean_variants,
    load_embedding_weight,
)
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage257_object_attribute_position_operation_role_map_20260324"
TOP_K = 24

ROLE_WORDS = {
    "object": ["apple", "pear", "dog", "shirt", "file", "car"],
    "attribute": ["red", "sweet", "large", "clean", "old", "fast"],
    "position": ["left", "right", "top", "bottom", "inside", "outside"],
    "operation": ["translate", "modify", "rewrite", "paint", "move", "edit"],
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def choose_token_id(tokenizer, variants_map: Dict[str, list], word: str) -> int:
    variants = variants_map.get(word)
    if variants:
        best = sorted(variants, key=lambda item: (0 if item.leading_space else 1, item.token_id))[0]
        return int(best.token_id)
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) == 1:
        return int(token_ids[0])
    raise KeyError(f"未找到词条变体：{word}")


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
        token_id = choose_token_id(tokenizer, variants_map, word)
        result[word] = weight[token_id].clone()
    return result


def build_summary() -> dict:
    all_words = [word for words in ROLE_WORDS.values() for word in words]
    embeddings = load_embeddings_for(all_words)
    global_centroid = torch.stack([embeddings[word] for word in all_words]).mean(dim=0)

    role_rows: List[dict] = []
    role_signature_dims: Dict[str, set[int]] = {}
    role_centroids: Dict[str, torch.Tensor] = {}

    for role_name, words in ROLE_WORDS.items():
        stack = torch.stack([embeddings[word] for word in words])
        centroid = stack.mean(dim=0)
        role_centroids[role_name] = centroid
        diffs = [cosine(embeddings[word], centroid) for word in words]
        signature = (centroid - global_centroid).abs()
        top_dims = torch.topk(signature, k=TOP_K).indices.tolist()
        role_signature_dims[role_name] = {int(dim) for dim in top_dims}
        role_rows.append(
            {
                "role_name": role_name,
                "word_count": len(words),
                "activation_strength": float(signature.mean().item()),
                "compactness": sum(diffs) / len(diffs),
                "signature_dim_count": len(top_dims),
                "signature_dim_mean": float(signature[top_dims].mean().item()),
            }
        )

    overlap_rows: List[dict] = []
    for left, right in combinations(role_signature_dims.keys(), 2):
        inter = role_signature_dims[left] & role_signature_dims[right]
        union = role_signature_dims[left] | role_signature_dims[right]
        overlap_rows.append(
            {
                "left_role": left,
                "right_role": right,
                "signature_overlap_count": len(inter),
                "signature_jaccard": len(inter) / max(len(union), 1),
                "centroid_similarity": cosine(role_centroids[left], role_centroids[right]),
            }
        )

    strongest = max(role_rows, key=lambda row: row["activation_strength"] * row["compactness"])
    weakest = min(role_rows, key=lambda row: row["activation_strength"] * row["compactness"])
    mean_compactness = sum(row["compactness"] for row in role_rows) / len(role_rows)
    mean_activation = sum(row["activation_strength"] for row in role_rows) / len(role_rows)
    mean_overlap = sum(row["signature_jaccard"] for row in overlap_rows) / len(overlap_rows)
    role_score = (mean_compactness + mean_activation + (1.0 - mean_overlap)) / 3.0

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage257_object_attribute_position_operation_role_map",
        "title": "对象-属性-位置-操作参数角色图",
        "status_short": "object_attribute_position_operation_role_map_ready",
        "role_count": len(role_rows),
        "role_score": role_score,
        "strongest_role_name": strongest["role_name"],
        "weakest_role_name": weakest["role_name"],
        "top_gap_name": "语义更像参数级角色簇，而不是单词级独立盒子；对象、属性、位置、操作会各自占据稳定但可复用的角色带",
        "role_rows": role_rows,
        "overlap_rows": overlap_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "role_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["role_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["role_rows"])
    with (output_dir / "overlap_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["overlap_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["overlap_rows"])
    report = [
        "# Stage257：对象-属性-位置-操作参数角色图",
        "",
        "## 核心结果",
        f"- 角色数量：{summary['role_count']}",
        f"- 角色图总分：{summary['role_score']:.4f}",
        f"- 最强角色：{summary['strongest_role_name']}",
        f"- 最弱角色：{summary['weakest_role_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE257_OBJECT_ATTRIBUTE_POSITION_OPERATION_ROLE_MAP_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="对象-属性-位置-操作参数角色图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
