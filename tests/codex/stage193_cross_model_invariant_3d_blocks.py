#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage193_cross_model_invariant_3d_blocks_20260323"

STAGE181_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage181_cross_model_shared_puzzle_board_20260323" / "summary.json"
STAGE187_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage187_cross_model_shared_puzzle_strengthening_20260323" / "summary.json"
STAGE190_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage190_cross_model_neuron_role_consensus_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_invariance(score: float) -> str:
    if score >= 0.7:
        return "不变稳定块"
    if score >= 0.5:
        return "不变过渡块"
    return "不变薄弱块"


def build_summary() -> dict:
    s181 = load_json(STAGE181_SUMMARY_PATH)
    s187 = load_json(STAGE187_SUMMARY_PATH)
    s190 = load_json(STAGE190_SUMMARY_PATH)

    priority_map = {str(row["piece_name"]): str(row["priority"]) for row in s187["piece_rows"]}
    block_rows = []
    for row in s181["block_rows"]:
        block_name = str(row["block_name"])
        score = float(row["score"])
        block_rows.append(
            {
                "block_name": block_name,
                "score": score,
                "priority": priority_map.get(block_name, "持续观察"),
                "status": classify_invariance(score),
            }
        )
    stable_block_count = sum(1 for row in block_rows if str(row["status"]) == "不变稳定块")
    weak_block_count = sum(1 for row in block_rows if str(row["status"]) == "不变薄弱块")
    invariant_block_score = (
        float(s190["consensus_score"]) * 0.50
        + sum(float(row["score"]) for row in block_rows) / float(len(block_rows)) * 0.50
    )
    strongest_block_name = max(block_rows, key=lambda row: float(row["score"]))["block_name"]
    weakest_block_name = min(block_rows, key=lambda row: float(row["score"]))["block_name"]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage193_cross_model_invariant_3d_blocks",
        "title": "跨模型不变三维拼块",
        "status_short": "cross_model_invariant_3d_blocks_ready",
        "block_count": len(block_rows),
        "stable_block_count": stable_block_count,
        "weak_block_count": weak_block_count,
        "strongest_block_name": strongest_block_name,
        "weakest_block_name": weakest_block_name,
        "invariant_block_score": invariant_block_score,
        "block_rows": block_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage193：跨模型不变三维拼块",
        "",
        "## 核心结果",
        f"- 拼块数量：{summary['block_count']}",
        f"- 不变稳定块数量：{summary['stable_block_count']}",
        f"- 不变薄弱块数量：{summary['weak_block_count']}",
        f"- 最强不变块：{summary['strongest_block_name']}",
        f"- 最弱不变块：{summary['weakest_block_name']}",
        f"- 不变拼块总分：{summary['invariant_block_score']:.4f}",
    ]
    (output_dir / "STAGE193_CROSS_MODEL_INVARIANT_3D_BLOCKS_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型不变三维拼块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
