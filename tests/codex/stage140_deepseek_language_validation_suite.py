#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import qwen3_language_shared as shared
import stage139_qwen3_language_validation_suite as qwen_suite


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323"


def deep_replace(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: deep_replace(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [deep_replace(item) for item in payload]
    if isinstance(payload, str):
        return (
            payload.replace("stage139_qwen3", "stage140_deepseek")
            .replace("stage139", "stage140")
            .replace("Qwen3", "DeepSeek7B")
            .replace("qwen3", "deepseek7b")
            .replace("Qwen/Qwen3-4B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            .replace(str(shared.QWEN3_MODEL_PATH), str(DEEPSEEK_MODEL_PATH))
        )
    return payload


def configure_base_paths() -> None:
    shared.QWEN3_MODEL_PATH = DEEPSEEK_MODEL_PATH
    qwen_suite.QWEN3_MODEL_PATH = DEEPSEEK_MODEL_PATH
    qwen_suite.OUTPUT_DIR = OUTPUT_DIR
    qwen_suite.VOCAB_SUMMARY_PATH = OUTPUT_DIR / "deepseek_vocab_summary.json"
    qwen_suite.WORD_ROWS_JSONL_PATH = OUTPUT_DIR / "deepseek_word_rows.jsonl"
    qwen_suite.WORD_ROWS_CSV_PATH = OUTPUT_DIR / "deepseek_word_rows.csv"
    qwen_suite.SUMMARY_PATH = OUTPUT_DIR / "summary.json"
    qwen_suite.REPORT_PATH = OUTPUT_DIR / "STAGE140_DEEPSEEK_LANGUAGE_VALIDATION_SUITE_REPORT.md"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_outputs(summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "STAGE140_DEEPSEEK_LANGUAGE_VALIDATION_SUITE_REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(qwen_suite.build_report(summary), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    configure_base_paths()
    raw_summary = qwen_suite.run_analysis(output_dir=output_dir, force=force)
    summary = deep_replace(raw_summary)
    summary["experiment_id"] = "stage140_deepseek_language_validation_suite"
    summary["title"] = "DeepSeek7B 语言理论迁移验证套件"
    summary["status_short"] = "deepseek7b_language_validation_ready"
    summary["model_name"] = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    summary["model_path"] = str(DEEPSEEK_MODEL_PATH)
    write_outputs(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek7B 语言理论迁移验证套件")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存，强制重跑")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "transfer_verdict": summary["transfer_summary"]["transfer_verdict"],
                "theory_check_pass_rate": summary["transfer_summary"]["theory_check_pass_rate"],
                "conditional_field_formula": summary["field_summary"]["best_formula"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
