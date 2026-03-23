#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from cross_model_language_shared import PROJECT_ROOT, build_all_model_bundles, clamp01, corr_to_unit


OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage142_cross_model_anaphora_hardening_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE142_CROSS_MODEL_ANAPHORA_HARDENING_REPORT.md"


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return None


def model_anaphora_rows() -> List[Dict[str, object]]:
    bundles = build_all_model_bundles()
    rows: List[Dict[str, object]] = []

    gpt2 = bundles["gpt2"]["stage136"]
    rows.append(
        {
            "model_key": "gpt2",
            "display_name": "GPT-2",
            "early_pronoun_corr": float(gpt2["mean_noun_pronoun_early_corr"]),
            "early_ellipsis_corr": float(gpt2["mean_noun_ellipsis_early_corr"]),
            "late_pronoun_corr": float(gpt2["mean_noun_pronoun_late_corr"]),
            "late_ellipsis_corr": float(gpt2["mean_noun_ellipsis_late_corr"]),
            "pronoun_sign_consistency": float(gpt2["mean_pronoun_sign_consistency_rate"]),
            "ellipsis_sign_consistency": float(gpt2["mean_ellipsis_sign_consistency_rate"]),
            "anaphora_score": float(gpt2["anaphora_ellipsis_propagation_score"]),
            "family_rows": list(gpt2["family_rows"]),
        }
    )

    for model_key in ("qwen3", "deepseek7b"):
        bundle = bundles[model_key]
        anaphora = bundle["anaphora"]
        rows.append(
            {
                "model_key": model_key,
                "display_name": bundle["display_name"],
                "early_pronoun_corr": float(anaphora["noun_pronoun_early_corr"]),
                "early_ellipsis_corr": float(anaphora["noun_ellipsis_early_corr"]),
                "late_pronoun_corr": float(anaphora["noun_pronoun_late_corr"]),
                "late_ellipsis_corr": float(anaphora["noun_ellipsis_late_corr"]),
                "pronoun_sign_consistency": sum(float(row["pronoun_sign_consistency_rate"]) for row in anaphora["family_rows"]) / len(anaphora["family_rows"]),
                "ellipsis_sign_consistency": sum(float(row["ellipsis_sign_consistency_rate"]) for row in anaphora["family_rows"]) / len(anaphora["family_rows"]),
                "anaphora_score": float(anaphora["anaphora_ellipsis_score"]),
                "family_rows": list(anaphora["family_rows"]),
            }
        )

    for row in rows:
        row["early_mean_corr"] = 0.5 * (float(row["early_pronoun_corr"]) + float(row["early_ellipsis_corr"]))
        row["late_mean_corr"] = 0.5 * (float(row["late_pronoun_corr"]) + float(row["late_ellipsis_corr"]))
        row["late_rescue_gap"] = float(row["late_mean_corr"]) - float(row["early_mean_corr"])
        row["early_closure_strength"] = corr_to_unit(float(row["early_mean_corr"]))
        row["late_recovery_strength"] = 0.5 * corr_to_unit(float(row["late_mean_corr"])) + 0.25 * float(row["pronoun_sign_consistency"]) + 0.25 * float(row["ellipsis_sign_consistency"])
        row["late_rescue_hardening_score"] = 0.55 * clamp01(float(row["late_rescue_gap"]) + 0.5) + 0.45 * float(row["late_recovery_strength"])
    return rows


def build_family_alignment(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_model = {row["model_key"]: {family["family_name"]: family for family in row["family_rows"]} for row in rows}
    family_names = sorted(set(by_model["gpt2"]) & set(by_model["qwen3"]) & set(by_model["deepseek7b"]))
    out = []
    for family_name in family_names:
        late_rescue_values = []
        all_positive = True
        for model_key in ("gpt2", "qwen3", "deepseek7b"):
            family = by_model[model_key][family_name]
            early_mean = 0.5 * (float(family["noun_pronoun_early_corr"]) + float(family["noun_ellipsis_early_corr"]))
            late_mean = 0.5 * (float(family["noun_pronoun_late_corr"]) + float(family["noun_ellipsis_late_corr"]))
            late_rescue = late_mean - early_mean
            late_rescue_values.append(late_rescue)
            all_positive = all_positive and late_rescue > 0.0
        mean_rescue = sum(late_rescue_values) / len(late_rescue_values)
        spread = max(late_rescue_values) - min(late_rescue_values)
        out.append(
            {
                "family_name": family_name,
                "mean_late_rescue": mean_rescue,
                "late_rescue_spread": spread,
                "all_models_positive": all_positive,
                "family_hardening_score": 0.6 * clamp01(mean_rescue + 0.5) + 0.4 * clamp01(1.0 - spread),
            }
        )
    return out


def build_summary(model_rows: List[Dict[str, object]], family_rows: List[Dict[str, object]]) -> Dict[str, object]:
    mean_model_score = sum(float(row["late_rescue_hardening_score"]) for row in model_rows) / len(model_rows)
    positive_family_rate = sum(1.0 for row in family_rows if row["all_models_positive"]) / max(1, len(family_rows))
    mean_family_score = sum(float(row["family_hardening_score"]) for row in family_rows) / max(1, len(family_rows))
    verdict = "late_rescue_without_early_closure" if positive_family_rate >= 0.8 else "anaphora_transfer_still_fragile"
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage142_cross_model_anaphora_hardening",
        "title": "跨模型回指与省略强化块",
        "status_short": "cross_model_anaphora_hardening_ready",
        "model_count": len(model_rows),
        "family_count": len(family_rows),
        "mean_model_hardening_score": mean_model_score,
        "all_model_positive_late_rescue_family_rate": positive_family_rate,
        "mean_family_hardening_score": mean_family_score,
        "hardening_verdict": verdict,
        "model_rows": model_rows,
        "family_rows": family_rows,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage142: 跨模型回指与省略强化块",
        "",
        "## 核心结果",
        f"- 模型数量: {summary['model_count']}",
        f"- 家族数量: {summary['family_count']}",
        f"- 平均模型强化分数: {summary['mean_model_hardening_score']:.4f}",
        f"- 全模型正向晚期补救家族率: {summary['all_model_positive_late_rescue_family_rate']:.4f}",
        f"- 平均家族强化分数: {summary['mean_family_hardening_score']:.4f}",
        f"- 判定: {summary['hardening_verdict']}",
    ]
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force:
        cached = load_cached_summary(output_dir)
        if cached is not None:
            return cached
    model_rows = model_anaphora_rows()
    family_rows = build_family_alignment(model_rows)
    summary = build_summary(model_rows, family_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型回指与省略强化块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重算")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
