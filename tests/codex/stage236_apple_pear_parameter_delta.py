#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage236_apple_pear_parameter_delta_20260324"

STAGE154_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage154_apple_fruit_shared_core_20260323" / "summary.json"
STAGE188_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage188_apple_neuron_role_card_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def similarity_of(rows: list[dict], word: str) -> float:
    for row in rows:
        if str(row["word"]) == word:
            return float(row["similarity_to_apple"])
    raise KeyError(word)


def role_score(rows: list[dict], role: str) -> float:
    for row in rows:
        if str(row["role_name"]) == role:
            return float(row["score"])
    raise KeyError(role)


def build_summary() -> dict:
    s154 = load_json(STAGE154_SUMMARY_PATH)
    s188 = load_json(STAGE188_SUMMARY_PATH)

    selected = s154["selected_similarity_rows"]
    apple_pear = similarity_of(selected, "pear")
    apple_banana = similarity_of(selected, "banana")
    apple_orange = similarity_of(selected, "orange")
    family_margin = float(s154["family_margin"])

    shared_bundle = role_score(s188["role_rows"], "共享束")
    delta_bundle = role_score(s188["role_rows"], "差分束")
    route_bundle = role_score(s188["role_rows"], "路径束")

    piece_rows = [
        {"piece_name": "苹果-梨共享近度", "score": apple_pear},
        {"piece_name": "苹果-香蕉共享近度", "score": apple_banana},
        {"piece_name": "苹果-橙子歧义干扰", "score": apple_orange},
        {"piece_name": "家族边界余量", "score": family_margin},
        {"piece_name": "共享束参数支持", "score": shared_bundle},
        {"piece_name": "差分束参数支持", "score": delta_bundle},
        {"piece_name": "路径束参数支持", "score": route_bundle},
    ]
    ranked_rows = sorted(piece_rows, key=lambda row: float(row["score"]), reverse=True)
    delta_score = sum(float(row["score"]) for row in piece_rows) / len(piece_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage236_apple_pear_parameter_delta",
        "title": "苹果-梨子参数级差分图",
        "status_short": "apple_pear_parameter_delta_ready",
        "piece_count": len(piece_rows),
        "delta_score": delta_score,
        "strongest_piece_name": str(ranked_rows[0]["piece_name"]),
        "weakest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "苹果与梨子的差分边界仍然偏薄",
        "apple_pear_similarity": apple_pear,
        "apple_banana_similarity": apple_banana,
        "apple_orange_similarity": apple_orange,
        "piece_rows": piece_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage236：苹果-梨子参数级差分图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 差分总分：{summary['delta_score']:.4f}",
        f"- 苹果-梨子近度：{summary['apple_pear_similarity']:.4f}",
        f"- 苹果-香蕉近度：{summary['apple_banana_similarity']:.4f}",
        f"- 苹果-橙子干扰近度：{summary['apple_orange_similarity']:.4f}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE236_APPLE_PEAR_PARAMETER_DELTA_REPORT.md").write_text(
        "\n".join(lines),
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
    parser = argparse.ArgumentParser(description="苹果-梨子参数级差分图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
