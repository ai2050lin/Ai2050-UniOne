#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage190_cross_model_neuron_role_consensus_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE190_CROSS_MODEL_NEURON_ROLE_CONSENSUS_REPORT.md"

STAGE159_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage159_triple_model_apple_kernel_20260323" / "summary.json"
STAGE181_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage181_cross_model_shared_puzzle_board_20260323" / "summary.json"
STAGE187_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage187_cross_model_shared_puzzle_strengthening_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_consensus(score: float, priority: str) -> str:
    if score >= 0.7:
        return "共同稳定角色"
    if priority == "优先补强":
        return "共同薄弱角色"
    return "共同过渡角色"


def build_summary() -> dict:
    s159 = load_json(STAGE159_SUMMARY_PATH)
    s181 = load_json(STAGE181_SUMMARY_PATH)
    s187 = load_json(STAGE187_SUMMARY_PATH)

    priority_map = {str(row["piece_name"]): str(row["priority"]) for row in s187["piece_rows"]}
    shared_map = {str(row["block_name"]): float(row["score"]) for row in s181["block_rows"]}

    role_rows = [
        {
            "role_name": "共享核角色",
            "score": float(s159["shared_core_consensus_score"]),
            "priority": "持续观察",
        },
        {
            "role_name": "条件场角色",
            "score": float(shared_map["条件场"]),
            "priority": priority_map["条件场"],
        },
        {
            "role_name": "副词动态选路角色",
            "score": float(shared_map["副词动态选路"]),
            "priority": priority_map["副词动态选路"],
        },
        {
            "role_name": "复杂语篇重提角色",
            "score": float(shared_map["复杂语篇重提"]),
            "priority": priority_map["复杂语篇重提"],
        },
        {
            "role_name": "结果链角色",
            "score": float(shared_map["结果链"]),
            "priority": priority_map["结果链"],
        },
    ]
    for row in role_rows:
        row["status"] = classify_consensus(float(row["score"]), str(row["priority"]))
    ranked_rows = sorted(role_rows, key=lambda row: float(row["score"]))
    stable_consensus_count = sum(1 for row in role_rows if str(row["status"]) == "共同稳定角色")
    weak_consensus_count = sum(1 for row in role_rows if str(row["status"]) == "共同薄弱角色")
    consensus_score = sum(float(row["score"]) for row in role_rows) / float(len(role_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage190_cross_model_neuron_role_consensus",
        "title": "跨模型神经元角色共同核",
        "status_short": "cross_model_neuron_role_consensus_ready",
        "role_count": len(role_rows),
        "stable_consensus_count": stable_consensus_count,
        "weak_consensus_count": weak_consensus_count,
        "strongest_role_name": str(ranked_rows[-1]["role_name"]),
        "weakest_role_name": str(ranked_rows[0]["role_name"]),
        "consensus_score": consensus_score,
        "role_rows": role_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage190：跨模型神经元角色共同核",
        "",
        "## 核心结果",
        f"- 角色数量：{summary['role_count']}",
        f"- 共同稳定角色数量：{summary['stable_consensus_count']}",
        f"- 共同薄弱角色数量：{summary['weak_consensus_count']}",
        f"- 最强共同角色：{summary['strongest_role_name']}",
        f"- 最弱共同角色：{summary['weakest_role_name']}",
        f"- 共同核总分：{summary['consensus_score']:.4f}",
    ]
    (output_dir / "STAGE190_CROSS_MODEL_NEURON_ROLE_CONSENSUS_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="跨模型神经元角色共同核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
