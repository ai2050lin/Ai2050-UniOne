#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage188_apple_neuron_role_card_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE188_APPLE_NEURON_ROLE_CARD_REPORT.md"

STAGE157_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json"
STAGE159_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage159_triple_model_apple_kernel_20260323" / "summary.json"
STAGE165_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage165_apple_to_language_kernel_bridge_20260323" / "summary.json"
STAGE172_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage172_provenance_trace_probe_20260323" / "summary.json"
STAGE186_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage186_result_binding_system_expansion_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_role(score: float) -> str:
    if score >= 0.75:
        return "稳定角色"
    if score >= 0.55:
        return "过渡角色"
    return "薄弱角色"


def find_piece_score(piece_rows: list[dict], piece_name: str) -> float:
    for row in piece_rows:
        if str(row["piece_name"]) == piece_name:
            return float(row["score"])
    raise KeyError(piece_name)


def build_summary() -> dict:
    s157 = load_json(STAGE157_SUMMARY_PATH)
    s159 = load_json(STAGE159_SUMMARY_PATH)
    s165 = load_json(STAGE165_SUMMARY_PATH)
    s172 = load_json(STAGE172_SUMMARY_PATH)
    s186 = load_json(STAGE186_SUMMARY_PATH)

    shared_score = float(s159["shared_core_consensus_score"])
    route_score = float(s157["apple_action_route_score"])
    trace_score = float(s172["provenance_trace_score"])
    binding_score = find_piece_score(s186["piece_rows"], "原生绑定")
    repair_score = find_piece_score(s186["piece_rows"], "修复能力")

    bridge_map = {str(row["proxy_name"]): float(row["score"]) for row in s165["proxy_rows"]}
    role_rows = [
        {
            "role_name": "共享束",
            "score": shared_score,
            "status": classify_role(shared_score),
            "evidence_anchor": "苹果共同核",
        },
        {
            "role_name": "差分束",
            "score": float(bridge_map["a_proxy"]),
            "status": classify_role(float(bridge_map["a_proxy"])),
            "evidence_anchor": "对象局部锚定",
        },
        {
            "role_name": "纤维束",
            "score": float(bridge_map["f_proxy"]),
            "status": classify_role(float(bridge_map["f_proxy"])),
            "evidence_anchor": "共享纤维参与",
        },
        {
            "role_name": "路径束",
            "score": route_score,
            "status": classify_role(route_score),
            "evidence_anchor": "动作选路",
        },
        {
            "role_name": "来源痕迹束",
            "score": trace_score,
            "status": classify_role(trace_score),
            "evidence_anchor": "来源痕迹",
        },
        {
            "role_name": "回收束",
            "score": binding_score,
            "status": classify_role(binding_score),
            "evidence_anchor": "原生绑定",
        },
        {
            "role_name": "修复束",
            "score": repair_score,
            "status": classify_role(repair_score),
            "evidence_anchor": "结果修复",
        },
    ]
    ranked_rows = sorted(role_rows, key=lambda row: float(row["score"]))
    weak_role_count = sum(1 for row in role_rows if str(row["status"]) == "薄弱角色")
    stable_role_count = sum(1 for row in role_rows if str(row["status"]) == "稳定角色")
    role_card_score = sum(float(row["score"]) for row in role_rows) / float(len(role_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage188_apple_neuron_role_card",
        "title": "苹果神经元角色卡",
        "status_short": "apple_neuron_role_card_ready",
        "role_count": len(role_rows),
        "stable_role_count": stable_role_count,
        "weak_role_count": weak_role_count,
        "strongest_role_name": str(ranked_rows[-1]["role_name"]),
        "weakest_role_name": str(ranked_rows[0]["role_name"]),
        "role_card_score": role_card_score,
        "role_rows": role_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage188：苹果神经元角色卡",
        "",
        "## 核心结果",
        f"- 角色数量：{summary['role_count']}",
        f"- 稳定角色数量：{summary['stable_role_count']}",
        f"- 薄弱角色数量：{summary['weak_role_count']}",
        f"- 最强角色：{summary['strongest_role_name']}",
        f"- 最弱角色：{summary['weakest_role_name']}",
        f"- 角色卡总分：{summary['role_card_score']:.4f}",
    ]
    (output_dir / "STAGE188_APPLE_NEURON_ROLE_CARD_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="苹果神经元角色卡")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
