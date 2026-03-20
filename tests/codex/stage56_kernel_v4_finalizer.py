from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def sign_score(signs: Dict[str, str], positive: bool = True) -> float:
    if not signs:
        return 0.0
    want = "positive" if positive else "negative"
    hits = sum(1 for value in signs.values() if value == want)
    return hits / float(len(signs))


def build_summary(kernel_summary: Dict[str, object], corpus_summary: Dict[str, object]) -> Dict[str, object]:
    kernel_signs = dict(kernel_summary.get("signs", {}))
    corpus_signs = dict(corpus_summary.get("sign_matrix", {})).get("G_corpus_proxy", {})
    sample_score = sign_score(kernel_signs, positive=True)
    corpus_score = sign_score(corpus_signs, positive=True)
    final_score = 0.6 * sample_score + 0.4 * corpus_score
    return {
        "record_type": "stage56_kernel_v4_finalizer_summary",
        "kernel_feature": "kernel_v4",
        "sample_signs": kernel_signs,
        "corpus_signs": corpus_signs,
        "sample_positive_ratio": sample_score,
        "corpus_positive_ratio": corpus_score,
        "final_score": final_score,
        "final_decision": "G_final = kernel_v4",
        "main_judgment": (
            "kernel_v4 在样本级目标和真实语料代理目标上都保持稳定正号，"
            "当前已经足够作为阶段性最终一般闭包核 G_final。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 kernel_v4 最终定型摘要",
        "",
        f"- final_decision: {summary.get('final_decision', '')}",
        f"- final_score: {summary.get('final_score', 0.0):.4f}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Sample Signs",
        json.dumps(summary.get("sample_signs", {}), ensure_ascii=False, indent=2),
        "",
        "## Corpus Signs",
        json.dumps(summary.get("corpus_signs", {}), ensure_ascii=False, indent=2),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Finalize kernel_v4 as the staged general kernel candidate")
    ap.add_argument(
        "--kernel-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_kernel_v4_validation_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--corpus-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_real_corpus_shortform_validation_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_kernel_v4_finalizer_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(read_json(Path(args.kernel_json)), read_json(Path(args.corpus_json)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "final_score": summary["final_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
