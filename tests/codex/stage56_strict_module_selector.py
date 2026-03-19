from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_fullsample_regression_runner import read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def build_summary(strict_summary: Dict[str, object]) -> Dict[str, object]:
    names = list(strict_summary.get("feature_names", []))
    fits = list(strict_summary.get("fits", []))
    ranking: List[Dict[str, object]] = []
    for feature in names:
        weights = {fit.get("target_name", ""): safe_float(dict(fit.get("weights", {})).get(feature)) for fit in fits}
        score = (
            safe_float(weights.get("strict_positive_synergy"))
            - (abs(safe_float(weights.get("union_joint_adv"))) + abs(safe_float(weights.get("union_synergy_joint")))) / 2.0
        )
        ranking.append({"feature": feature, "selectivity_score": score, "weights": weights})
    ranking.sort(key=lambda item: safe_float(item.get("selectivity_score")), reverse=True)
    top_score = safe_float(ranking[0]["selectivity_score"]) if ranking else 0.0
    top_candidates = [
        item
        for item in ranking
        if top_score - safe_float(item.get("selectivity_score")) <= 0.01
    ]
    return {
        "record_type": "stage56_strict_module_selector_summary",
        "candidate_count": len(ranking),
        "ranking": ranking,
        "top_candidates": top_candidates,
        "main_judgment": (
            "严格模块候选的选择标准已经从“符号是否翻转”推进到“严格闭包选择性”比较，"
            "当前最需要看的不是谁单独最大，而是谁最能提升严格闭包、同时最少污染一般闭包。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 严格模块候选排序摘要",
        "",
        f"- candidate_count: {summary.get('candidate_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Top Candidates",
    ]
    for row in list(summary.get("top_candidates", [])):
        row = dict(row)
        lines.append(f"- {row.get('feature', '')}: score={safe_float(row.get('selectivity_score')):+.6f}")
    lines.extend(["", "## Ranking"])
    for row in list(summary.get("ranking", [])):
        row = dict(row)
        lines.append(f"- {row.get('feature', '')}: score={safe_float(row.get('selectivity_score')):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rank strict module candidates by strict-closure selectivity")
    ap.add_argument(
        "--strict-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_load_module_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_module_selector_20260319"),
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
    print(json.dumps({"output_dir": str(output_dir), "candidate_count": summary.get("candidate_count", 0)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
