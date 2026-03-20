from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    from tests.codex.stage56_language_online_injection_experiment import ROOT, run_experiment
except ModuleNotFoundError:
    from stage56_language_online_injection_experiment import ROOT, run_experiment  # type: ignore


def run_suite(
    corpus_path: Path,
    max_lines: int = 320,
    max_vocab: int = 768,
    short_ctx_len: int = 4,
    long_ctx_len: int = 8,
    base_epochs: int = 8,
    inject_steps: int = 16,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, object]:
    short_artifacts = run_experiment(
        corpus_path=corpus_path,
        max_lines=max_lines,
        max_vocab=max_vocab,
        ctx_len=short_ctx_len,
        base_epochs=base_epochs,
        inject_steps=inject_steps,
        batch_size=batch_size,
        seed=seed,
    )
    long_artifacts = run_experiment(
        corpus_path=corpus_path,
        max_lines=max_lines,
        max_vocab=max_vocab,
        ctx_len=long_ctx_len,
        base_epochs=base_epochs,
        inject_steps=inject_steps,
        batch_size=batch_size,
        seed=seed,
    )

    short_before_base = short_artifacts.summary["before_injection"]["base_valid"]
    short_after_base = short_artifacts.summary["after_injection"]["base_valid"]
    short_after_novel = short_artifacts.summary["after_injection"]["novel_valid"]
    long_before_base = long_artifacts.summary["before_injection"]["base_valid"]
    long_after_base = long_artifacts.summary["after_injection"]["base_valid"]
    long_after_novel = long_artifacts.summary["after_injection"]["novel_valid"]

    summary = {
        "record_type": "stage56_long_context_online_language_suite_summary",
        "config": {
            "corpus_path": str(corpus_path),
            "max_lines": max_lines,
            "max_vocab": max_vocab,
            "short_ctx_len": short_ctx_len,
            "long_ctx_len": long_ctx_len,
            "base_epochs": base_epochs,
            "inject_steps": inject_steps,
            "batch_size": batch_size,
            "seed": seed,
        },
        "short_context": short_artifacts.summary,
        "long_context": long_artifacts.summary,
        "comparison": {
            "short_base_perplexity_delta": short_after_base["perplexity"] - short_before_base["perplexity"],
            "long_base_perplexity_delta": long_after_base["perplexity"] - long_before_base["perplexity"],
            "short_novel_accuracy_after": short_after_novel["accuracy"],
            "long_novel_accuracy_after": long_after_novel["accuracy"],
            "short_forgetting": short_artifacts.summary["deltas"]["forgetting"],
            "long_forgetting": long_artifacts.summary["deltas"]["forgetting"],
        },
        "main_judgment": (
            "更长上下文语言任务会放大在线注入后的分布漂移压力；"
            "如果长上下文下 novel 提升仍能成立，而 base 困惑度恶化更明显，就说明当前原型已经开始触到真实长期语言学习里的稳定性-可塑性冲突。"
        ),
    }
    return summary


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 长上下文语言任务与更长期在线注入套件",
            "",
            f"- main_judgment: {summary['main_judgment']}",
            "",
            "## Comparison",
            json.dumps(summary["comparison"], ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a longer-horizon and longer-context online language suite")
    ap.add_argument("--corpus-path", default=str(ROOT / "tempdata" / "wiki_train.txt"))
    ap.add_argument("--max-lines", type=int, default=320)
    ap.add_argument("--max-vocab", type=int, default=768)
    ap.add_argument("--short-ctx-len", type=int, default=4)
    ap.add_argument("--long-ctx-len", type=int, default=8)
    ap.add_argument("--base-epochs", type=int, default=8)
    ap.add_argument("--inject-steps", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_long_context_online_language_suite_20260320"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_suite(
        corpus_path=Path(args.corpus_path),
        max_lines=args.max_lines,
        max_vocab=args.max_vocab,
        short_ctx_len=args.short_ctx_len,
        long_ctx_len=args.long_ctx_len,
        base_epochs=args.base_epochs,
        inject_steps=args.inject_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
