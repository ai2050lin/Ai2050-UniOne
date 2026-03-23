#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import torch
from safetensors import safe_open

from qwen3_language_shared import QWEN3_MODEL_PATH, load_qwen3_embedding_weight
from stage119_gpt2_embedding_full_vocab_scan import MODEL_PATH as GPT2_MODEL_PATH
from stage119_gpt2_embedding_full_vocab_scan import load_embedding_weight as load_gpt2_embedding_weight


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage159_triple_model_apple_kernel_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE159_TRIPLE_MODEL_APPLE_KERNEL_REPORT.md"

GPT2_ROWS_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323" / "word_rows.jsonl"
QWEN_ROWS_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323" / "qwen_word_rows.jsonl"
DEEPSEEK_ROWS_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "deepseek_word_rows.jsonl"
DEEPSEEK_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def l2_unit(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp_min(1e-8)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(l2_unit(a.float()), l2_unit(b.float())).item())


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def load_deepseek_embedding_weight() -> torch.Tensor:
    for shard_path in sorted(DEEPSEEK_MODEL_PATH.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            if "model.embed_tokens.weight" in handle.keys():
                return handle.get_tensor("model.embed_tokens.weight").detach().cpu()
    raise FileNotFoundError("未在 DeepSeek safetensors 中找到 model.embed_tokens.weight")


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
        scored.append({"word": str(row["word"]), "group": str(row["group"]), "similarity": score})
    scored.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return scored[:count]


def build_model_row(model_name: str, rows: Sequence[Dict[str, object]], embed_weight: torch.Tensor) -> Dict[str, object]:
    word_map = {str(row["word"]).lower(): row for row in rows}
    apple_row = word_map["apple"]
    apple_vec = embed_weight[int(apple_row["token_id"])]
    fruit_rows = [
        row
        for row in rows
        if str(row["lexical_type"]) == "noun" and str(row["group"]) == "meso_fruit"
    ]
    nonfruit_rows = [
        row
        for row in rows
        if str(row["lexical_type"]) == "noun" and str(row["group"]) != "meso_fruit"
    ]
    fruit_centroid = l2_unit(torch.stack([embed_weight[int(row["token_id"])] for row in fruit_rows], dim=0).mean(dim=0))
    nonfruit_centroid = l2_unit(
        torch.stack([embed_weight[int(row["token_id"])] for row in nonfruit_rows[:4096]], dim=0).mean(dim=0)
    )
    top_fruit_neighbors = top_neighbors(apple_vec, fruit_rows, embed_weight, skip_words=["apple"], count=8)
    top_contrast_neighbors = top_neighbors(apple_vec, nonfruit_rows, embed_weight, skip_words=["apple"], count=8)
    apple_to_fruit_centroid = cosine(apple_vec, fruit_centroid)
    apple_to_nonfruit_centroid = cosine(apple_vec, nonfruit_centroid)
    top_fruit_neighbor_mean = mean(float(item["similarity"]) for item in top_fruit_neighbors)
    family_margin = apple_to_fruit_centroid - apple_to_nonfruit_centroid
    shared_core_score = clamp01(
        0.40 * ((apple_to_fruit_centroid + 1.0) / 2.0)
        + 0.30 * ((top_fruit_neighbor_mean + 1.0) / 2.0)
        + 0.30 * ((family_margin + 1.0) / 2.0)
    )
    return {
        "model_name": model_name,
        "apple_group": str(apple_row["group"]),
        "apple_band": str(apple_row["band"]),
        "fruit_member_count": len(fruit_rows),
        "apple_to_fruit_centroid": apple_to_fruit_centroid,
        "apple_to_nonfruit_centroid": apple_to_nonfruit_centroid,
        "family_margin": family_margin,
        "shared_core_score": shared_core_score,
        "top_fruit_neighbors": top_fruit_neighbors,
        "top_contrast_neighbors": top_contrast_neighbors,
    }


def build_summary() -> Dict[str, object]:
    gpt2_row = build_model_row("GPT-2", load_rows(GPT2_ROWS_PATH), load_gpt2_embedding_weight())
    qwen_row = build_model_row("Qwen3-4B", load_rows(QWEN_ROWS_PATH), load_qwen3_embedding_weight())
    deepseek_row = build_model_row("DeepSeek-R1-Distill-Qwen-7B", load_rows(DEEPSEEK_ROWS_PATH), load_deepseek_embedding_weight())
    model_rows = [gpt2_row, qwen_row, deepseek_row]
    strongest = max(model_rows, key=lambda row: float(row["shared_core_score"]))
    weakest = min(model_rows, key=lambda row: float(row["shared_core_score"]))
    neighbor_sets = [set(item["word"] for item in row["top_fruit_neighbors"]) for row in model_rows]
    common_neighbor_words = sorted(set.intersection(*neighbor_sets))
    shared_core_consensus_score = clamp01(
        0.70 * mean(float(row["shared_core_score"]) for row in model_rows)
        + 0.30 * min(1.0, len(common_neighbor_words) / 4.0)
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage159_triple_model_apple_kernel",
        "title": "苹果三模型共同核",
        "status_short": "triple_model_apple_kernel_ready",
        "model_count": len(model_rows),
        "common_neighbor_words": common_neighbor_words,
        "common_neighbor_overlap_count": len(common_neighbor_words),
        "shared_core_consensus_score": shared_core_consensus_score,
        "strongest_model_name": str(strongest["model_name"]),
        "weakest_model_name": str(weakest["model_name"]),
        "model_rows": model_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage159: 苹果三模型共同核",
        "",
        "## 核心结果",
        f"- 模型数: {summary['model_count']}",
        f"- 共同近邻词数: {summary['common_neighbor_overlap_count']}",
        f"- 共同核分数: {summary['shared_core_consensus_score']:.4f}",
        f"- 最强模型: {summary['strongest_model_name']}",
        f"- 最弱模型: {summary['weakest_model_name']}",
        f"- 共同近邻词: {', '.join(summary['common_neighbor_words'])}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果三模型共同核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
