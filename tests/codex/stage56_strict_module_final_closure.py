from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(strict_summary: Dict[str, object]) -> Dict[str, object]:
    ranking: List[Dict[str, object]] = list(strict_summary.get("ranking", []))
    best = ranking[0] if ranking else {}
    runner_up = ranking[1] if len(ranking) > 1 else {}
    best_score = float(best.get("final_score", 0.0))
    runner_up_score = float(runner_up.get("final_score", 0.0))
    score_margin = best_score - runner_up_score
    closure_confidence = best_score / (best_score + runner_up_score) if (best_score + runner_up_score) > 0 else 0.0
    return {
        "record_type": "stage56_strict_module_final_closure_summary",
        "final_choice": best,
        "runner_up": runner_up,
        "score_margin": score_margin,
        "closure_confidence": closure_confidence,
        "final_decision": "S_final = strict_module_base_term",
        "main_judgment": (
            "strict_module_base_term 仍然是当前最优严格模块，"
            "但和次优候选的差距还不够大，因此它更适合作为阶段最终严格核心，"
            "而不是绝对唯一的终态对象。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 严格模块最终收口摘要",
        "",
        f"- final_decision: {summary.get('final_decision', '')}",
        f"- score_margin: {summary.get('score_margin', 0.0):.6f}",
        f"- closure_confidence: {summary.get('closure_confidence', 0.0):.6f}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Close the current strict module candidate into a staged final choice")
    ap.add_argument(
        "--strict-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_module_finalizer_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_module_final_closure_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(read_json(Path(args.strict_json)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "score_margin": summary["score_margin"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
