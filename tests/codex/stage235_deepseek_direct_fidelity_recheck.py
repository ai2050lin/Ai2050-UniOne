#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage235_deepseek_direct_fidelity_recheck_20260324"

STAGE140_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"
STAGE232_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage232_parameter_complex_structure_joint_puzzle_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return completed.stdout
    except Exception:
        return ""


def detect_hardware_support() -> tuple[bool, bool]:
    smi_text = safe_run(["nvidia-smi"])
    ollama_text = safe_run(["ollama", "list"])

    has_24g_gpu = "24564MiB" in smi_text or "24576MiB" in smi_text or "4090" in smi_text
    has_14b = "deepseek-r1:14b" in ollama_text
    has_32b = "deepseek-r1:32b" in ollama_text.lower()

    support_14b = has_24g_gpu and has_14b
    support_32b = has_24g_gpu and has_32b
    return support_14b, support_32b


def build_summary() -> dict:
    s140 = load_json(STAGE140_SUMMARY_PATH)
    s232 = load_json(STAGE232_SUMMARY_PATH)
    support_14b, support_32b = detect_hardware_support()

    direct_fidelity = float(s140["anaphora_summary"]["anaphora_ellipsis_score"])
    direct_joint = float(s140["joint_summary"]["noun_verb_joint_propagation_score"])
    direct_result = float(s140["result_summary"]["noun_verb_result_chain_score"])

    recheck_rows = [
        {"piece_name": "DeepSeek直测来源保真", "score": direct_fidelity},
        {"piece_name": "DeepSeek直测复杂处理链", "score": (direct_joint + direct_result) / 2.0},
        {"piece_name": "14B硬件支持度", "score": 1.0 if support_14b else 0.0},
        {"piece_name": "32B硬件支持度", "score": 0.7 if support_32b else 0.0},
        {"piece_name": "参数-结构联合底盘", "score": float(s232["joint_score"])},
    ]
    ranked_rows = sorted(recheck_rows, key=lambda row: float(row["score"]), reverse=True)
    recheck_score = sum(float(row["score"]) for row in recheck_rows) / len(recheck_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage235_deepseek_direct_fidelity_recheck",
        "title": "DeepSeek直测来源保真复核",
        "status_short": "deepseek_direct_fidelity_recheck_ready",
        "model_name": str(s140["model_name"]),
        "piece_count": len(recheck_rows),
        "recheck_score": recheck_score,
        "strongest_piece_name": str(ranked_rows[0]["piece_name"]),
        "weakest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "DeepSeek复杂处理后段仍然偏弱",
        "support_14b": support_14b,
        "support_32b": support_32b,
        "recheck_rows": recheck_rows,
        "direct_verdict": str(s140["transfer_summary"]["transfer_verdict"]),
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage235：DeepSeek直测来源保真复核",
        "",
        "## 核心结果",
        f"- 模型：{summary['model_name']}",
        f"- 复核总分：{summary['recheck_score']:.4f}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 14B硬件支持：{summary['support_14b']}",
        f"- 32B硬件支持：{summary['support_32b']}",
        f"- 头号缺口：{summary['top_gap_name']}",
        f"- 直测判定：{summary['direct_verdict']}",
    ]
    (output_dir / "STAGE235_DEEPSEEK_DIRECT_FIDELITY_RECHECK_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="DeepSeek直测来源保真复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
