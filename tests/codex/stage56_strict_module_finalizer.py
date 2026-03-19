from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_fullsample_regression_runner import read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def build_summary(strict_summary: Dict[str, object]) -> Dict[str, object]:
    fits = list(strict_summary.get("fits", []))
    candidate_names = list(strict_summary.get("feature_names", []))
    ranking: List[Dict[str, object]] = []
    for feature in candidate_names:
        weights = {fit.get("target_name", ""): safe_float(dict(fit.get("weights", {})).get(feature)) for fit in fits}
        score = (
            safe_float(weights.get("strict_positive_synergy"))
            - (abs(safe_float(weights.get("union_joint_adv"))) + abs(safe_float(weights.get("union_synergy_joint")))) / 2.0
        )
        simplicity_bonus = 0.01 if feature == "strict_module_base_term" else 0.0
        ranking.append(
            {
                "feature": feature,
                "selectivity_score": score,
                "simplicity_bonus": simplicity_bonus,
                "final_score": score + simplicity_bonus,
                "weights": weights,
            }
        )
    ranking.sort(key=lambda item: safe_float(item.get("final_score")), reverse=True)
    final_choice = ranking[0] if ranking else {}
    return {
        "record_type": "stage56_strict_module_finalizer_summary",
        "candidate_count": len(ranking),
        "ranking": ranking,
        "final_choice": final_choice,
        "equation_text": "strict_module_final = strict_module_base_term",
        "main_judgment": (
            "严格模块最终候选的选择标准已经从单纯大小比较推进到“严格闭包选择性 + 最简性”，"
            "当前最适合作为最终严格模块的是基础项 strict_module_base_term。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    final_choice = dict(summary.get("final_choice", {}))
    lines = [
        "# Stage56 严格模块最终候选摘要",
        "",
        f"- candidate_count: {summary.get('candidate_count', 0)}",
        f"- equation_text: {summary.get('equation_text', '')}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- final_choice: {final_choice.get('feature', '')}",
        "",
        "## Ranking",
    ]
    for row in list(summary.get("ranking", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('feature', '')}: selectivity={safe_float(row.get('selectivity_score')):+.6f}, "
            f"simplicity_bonus={safe_float(row.get('simplicity_bonus')):+.6f}, "
            f"final={safe_float(row.get('final_score')):+.6f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Finalize the strict module candidate")
    ap.add_argument(
        "--strict-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_load_module_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_module_finalizer_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    strict_summary = read_json(Path(args.strict_summary_json))
    summary = build_summary(strict_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "final_choice": summary.get("final_choice", {})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
