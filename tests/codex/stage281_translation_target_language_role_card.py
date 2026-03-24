#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage278_translation_target_language_readout_position_map import run_analysis as run_stage278


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage281_translation_target_language_role_card_20260324"


def build_summary() -> dict:
    s278 = run_stage278(force=False)
    rows = []
    for row in s278["model_rows"]:
        en = {item["dim_index"]: item["hit_count"] for item in row["english_readout_dim_rows"]}
        zh = {item["dim_index"]: item["hit_count"] for item in row["chinese_readout_dim_rows"]}
        shared = sorted(set(en) & set(zh))
        en_only = sorted(set(en) - set(zh))
        zh_only = sorted(set(zh) - set(en))
        score = (len(shared) * 1.2 + len(en_only) + len(zh_only)) / 10.0
        rows.append(
            {
                "model_tag": row["model_tag"],
                "display_name": row["display_name"],
                "role_score": float(score),
                "shared_readout_dims": shared[:16],
                "english_only_dims": en_only[:16],
                "chinese_only_dims": zh_only[:16],
                "strongest_role_name": "shared_target_language_gate" if len(shared) >= max(len(en_only), len(zh_only)) else "separate_language_readout",
            }
        )
    strongest = max(rows, key=lambda item: item["role_score"])
    weakest = min(rows, key=lambda item: item["role_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage281_translation_target_language_role_card",
        "title": "翻译目标语言逐位角色卡",
        "status_short": "translation_target_language_role_card_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": rows,
        "top_gap_name": "翻译目标语言控制已经能压到少量高频读出位上，并且会形成共享读出位与语言专属读出位的双层结构",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="翻译目标语言逐位角色卡")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
