#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import torch

from stage119_gpt2_embedding_full_vocab_scan import load_embedding_weight
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE119_OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage154_apple_fruit_shared_core_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE154_APPLE_FRUIT_SHARED_CORE_REPORT.md"

TARGET_FRUITS = ["apple", "orange", "banana", "pear", "peach", "grape", "melon", "berry", "fruit"]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def l2_unit(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp_min(1e-8)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(l2_unit(a.float()), l2_unit(b.float())).item())


def top_neighbors(
    target_vec: torch.Tensor,
    rows: Sequence[Dict[str, object]],
    embed_weight: torch.Tensor,
    *,
    skip_words: Sequence[str],
    count: int,
) -> List[Dict[str, object]]:
    scored: List[Dict[str, object]] = []
    skip = {word.lower() for word in skip_words}
    target_unit = l2_unit(target_vec.float())
    for row in rows:
        word = str(row["word"]).lower()
        if word in skip:
            continue
        vec = embed_weight[int(row["token_id"])]
        score = float(torch.dot(target_unit, l2_unit(vec.float())).item())
        scored.append(
            {
                "word": str(row["word"]),
                "group": str(row["group"]),
                "band": str(row["band"]),
                "lexical_type": str(row["lexical_type"]),
                "similarity": score,
            }
        )
    scored.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return scored[:count]


def build_summary(rows: Sequence[Dict[str, object]], embed_weight: torch.Tensor) -> Dict[str, object]:
    word_map = {str(row["word"]).lower(): row for row in rows}
    apple_row = word_map["apple"]
    apple_vec = embed_weight[int(apple_row["token_id"])]

    fruit_rows = [
        row
        for row in rows
        if str(row["lexical_type"]) == "noun" and str(row["group"]) == "meso_fruit"
    ]
    nonfruit_noun_rows = [
        row
        for row in rows
        if str(row["lexical_type"]) == "noun" and str(row["group"]) != "meso_fruit"
    ]
    fruit_rows_sorted = sorted(
        fruit_rows,
        key=lambda row: (
            float(row.get("effective_encoding_score", 0.0)),
            float(row.get("lexical_type_score", 0.0)),
        ),
        reverse=True,
    )

    fruit_vecs = torch.stack([embed_weight[int(row["token_id"])] for row in fruit_rows], dim=0)
    fruit_centroid = l2_unit(fruit_vecs.mean(dim=0))
    nonfruit_centroid = l2_unit(
        torch.stack([embed_weight[int(row["token_id"])] for row in nonfruit_noun_rows[:4096]], dim=0).mean(dim=0)
    )

    selected_rows = [word_map[word] for word in TARGET_FRUITS if word in word_map]
    selected_similarity_rows = [
        {
            "word": str(row["word"]),
            "group": str(row["group"]),
            "similarity_to_apple": cosine(apple_vec, embed_weight[int(row["token_id"])]),
        }
        for row in selected_rows
    ]
    top_fruit_neighbors = top_neighbors(apple_vec, fruit_rows_sorted, embed_weight, skip_words=["apple"], count=8)
    top_contrast_neighbors = top_neighbors(apple_vec, nonfruit_noun_rows, embed_weight, skip_words=["apple"], count=8)

    apple_to_fruit_centroid = cosine(apple_vec, fruit_centroid)
    apple_to_nonfruit_centroid = cosine(apple_vec, nonfruit_centroid)
    top_fruit_neighbor_mean = mean(float(item["similarity"]) for item in top_fruit_neighbors)
    top_contrast_neighbor_mean = mean(float(item["similarity"]) for item in top_contrast_neighbors)
    family_margin = apple_to_fruit_centroid - apple_to_nonfruit_centroid
    neighbor_margin = top_fruit_neighbor_mean - top_contrast_neighbor_mean
    shared_core_score = clamp01(
        0.40 * ((apple_to_fruit_centroid + 1.0) / 2.0)
        + 0.30 * ((top_fruit_neighbor_mean + 1.0) / 2.0)
        + 0.30 * ((family_margin + 1.0) / 2.0)
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage154_apple_fruit_shared_core",
        "title": "苹果-水果共享核分析",
        "status_short": "apple_fruit_shared_core_ready",
        "fruit_member_count": len(fruit_rows),
        "target_word": "apple",
        "apple_group": str(apple_row["group"]),
        "apple_band": str(apple_row["band"]),
        "apple_to_fruit_centroid": apple_to_fruit_centroid,
        "apple_to_nonfruit_centroid": apple_to_nonfruit_centroid,
        "top_fruit_neighbor_mean": top_fruit_neighbor_mean,
        "top_contrast_neighbor_mean": top_contrast_neighbor_mean,
        "family_margin": family_margin,
        "neighbor_margin": neighbor_margin,
        "shared_core_score": shared_core_score,
        "selected_similarity_rows": selected_similarity_rows,
        "top_fruit_neighbors": top_fruit_neighbors,
        "top_contrast_neighbors": top_contrast_neighbors,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage154: 苹果-水果共享核分析",
        "",
        "## 核心结果",
        f"- 水果成员数: {summary['fruit_member_count']}",
        f"- 苹果到水果质心相似度: {summary['apple_to_fruit_centroid']:.4f}",
        f"- 苹果到非水果名词质心相似度: {summary['apple_to_nonfruit_centroid']:.4f}",
        f"- 家族边距: {summary['family_margin']:.4f}",
        f"- 邻居边距: {summary['neighbor_margin']:.4f}",
        f"- 共享核分数: {summary['shared_core_score']:.4f}",
        "",
        "## 近邻",
    ]
    for row in summary["top_fruit_neighbors"]:
        lines.append(f"- 水果近邻 {row['word']}: {row['similarity']:.4f}")
    for row in summary["top_contrast_neighbors"][:5]:
        lines.append(f"- 对比近邻 {row['word']}: {row['similarity']:.4f}")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    _, rows = ensure_stage119_rows(STAGE119_OUTPUT_DIR)
    embed_weight = load_embedding_weight()
    summary = build_summary(rows, embed_weight)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果-水果共享核分析")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
