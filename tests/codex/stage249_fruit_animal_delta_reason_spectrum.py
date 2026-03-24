#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from stage119_gpt2_embedding_full_vocab_scan import (
    MODEL_PATH,
    collect_clean_variants,
    load_embedding_weight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage249_fruit_animal_delta_reason_spectrum_20260324"

FAMILIES = {
    "fruit": ["apple", "pear", "banana", "peach", "orange", "grape", "lemon", "melon"],
    "animal": ["cat", "dog", "lion", "tiger", "horse", "rabbit", "wolf", "bear"],
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


def load_word_embeddings() -> Dict[str, Dict[str, torch.Tensor]]:
    tokenizer = load_tokenizer()
    variants_map, _ = collect_clean_variants(tokenizer)
    weight = load_embedding_weight()
    result: Dict[str, Dict[str, torch.Tensor]] = {}
    for family_name, words in FAMILIES.items():
        result[family_name] = {}
        for word in words:
            result[family_name][word] = weight[choose_token_id(variants_map, word)].clone()
    return result


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def build_summary() -> dict:
    family_embeddings = load_word_embeddings()
    family_centroids = {
        family_name: torch.stack(list(word_map.values())).mean(dim=0)
        for family_name, word_map in family_embeddings.items()
    }
    all_words = [(family_name, word, embedding) for family_name, word_map in family_embeddings.items() for word, embedding in word_map.items()]

    word_rows: List[dict] = []
    family_rows: List[dict] = []
    for family_name, word_map in family_embeddings.items():
        family_word_rows = []
        for word, embedding in word_map.items():
            family_share = cosine(embedding, family_centroids[family_name])
            same_family_peers = [cosine(embedding, peer) for peer_word, peer in word_map.items() if peer_word != word]
            other_family_peers = [cosine(embedding, peer) for other_family, other_word, peer in all_words if other_family != family_name]
            intra_competition = max(same_family_peers)
            cross_category_pressure = max(other_family_peers)
            margin = intra_competition - cross_category_pressure
            if cross_category_pressure > 0.50:
                dominant_reason = "跨类别分离主导"
            elif intra_competition > 0.50 and margin < 0.12:
                dominant_reason = "家族内部竞争主导"
            else:
                dominant_reason = "共享基底主导"
            row = {
                "family_name": family_name,
                "word": word,
                "family_share": family_share,
                "intra_competition": intra_competition,
                "cross_category_pressure": cross_category_pressure,
                "margin": margin,
                "dominant_reason": dominant_reason,
            }
            family_word_rows.append(row)
            word_rows.append(row)
        family_rows.append(
            {
                "family_name": family_name,
                "mean_family_share": sum(row["family_share"] for row in family_word_rows) / len(family_word_rows),
                "mean_intra_competition": sum(row["intra_competition"] for row in family_word_rows) / len(family_word_rows),
                "mean_cross_category_pressure": sum(row["cross_category_pressure"] for row in family_word_rows) / len(family_word_rows),
                "mean_margin": sum(row["margin"] for row in family_word_rows) / len(family_word_rows),
            }
        )

    fruit_row = next(row for row in family_rows if row["family_name"] == "fruit")
    animal_row = next(row for row in family_rows if row["family_name"] == "animal")
    strongest = max(word_rows, key=lambda row: row["margin"])
    weakest = min(word_rows, key=lambda row: row["margin"])
    spectrum_score = (
        fruit_row["mean_family_share"]
        + animal_row["mean_family_share"]
        + (1.0 - fruit_row["mean_margin"])
        + animal_row["mean_margin"]
    ) / 4.0
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage249_fruit_animal_delta_reason_spectrum",
        "title": "水果-动物差分原因谱",
        "status_short": "fruit_animal_delta_reason_spectrum_ready",
        "word_count": len(word_rows),
        "family_count": len(family_rows),
        "fruit_mean_margin": fruit_row["mean_margin"],
        "animal_mean_margin": animal_row["mean_margin"],
        "fruit_mean_family_share": fruit_row["mean_family_share"],
        "animal_mean_family_share": animal_row["mean_family_share"],
        "spectrum_score": spectrum_score,
        "strongest_word_name": strongest["word"],
        "weakest_word_name": weakest["word"],
        "top_gap_name": "水果整体更偏共享基底主导，动物整体更偏边界差分主导",
        "family_rows": family_rows,
        "word_rows": word_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "word_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["word_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["word_rows"])
    report = [
        "# Stage249：水果-动物差分原因谱",
        "",
        "## 核心结果",
        f"- 词数量：{summary['word_count']}",
        f"- 水果平均边界余量：{summary['fruit_mean_margin']:.4f}",
        f"- 动物平均边界余量：{summary['animal_mean_margin']:.4f}",
        f"- 水果平均家族共享：{summary['fruit_mean_family_share']:.4f}",
        f"- 动物平均家族共享：{summary['animal_mean_family_share']:.4f}",
        f"- 原因谱总分：{summary['spectrum_score']:.4f}",
        f"- 最强词：{summary['strongest_word_name']}",
        f"- 最弱词：{summary['weakest_word_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE249_FRUIT_ANIMAL_DELTA_REASON_SPECTRUM_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="水果-动物差分原因谱")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
