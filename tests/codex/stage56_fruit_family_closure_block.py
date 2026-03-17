from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "tests" / "codex" / "stage56_multimodel_sequential_pipeline.py"
DEFAULT_ITEMS_FILE = ROOT / "tests" / "codex" / "stage56_fruit_family_closure_items.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "tempdata" / "stage56_fruit_family_closure_block"


def build_command(args: argparse.Namespace) -> List[str]:
    python_exe = args.python_exe or sys.executable
    command = [
        python_exe,
        str(PIPELINE),
        "--models",
        args.models,
        "--items-file",
        str(Path(args.items_file)),
        "--output-root",
        str(Path(args.output_root)),
        "--survey-per-category",
        "3",
        "--deep-per-category",
        "3",
        "--closure-per-category",
        "3",
        "--anchors-per-category",
        "1",
        "--challengers-per-category",
        "2",
        "--supports-per-category",
        "0",
        "--family-count",
        "4",
        "--terms-per-family",
        "3",
        "--shared-k",
        "8",
        "--specific-k",
        "4",
        "--signature-top-k",
        "64",
        "--subset-sizes",
        "8,4",
        "--stage5-max-candidates",
        "4",
        "--stage5-per-category-limit",
        "1",
        "--stage5-max-neurons-per-candidate",
        "6",
        "--stage5-max-neurons-per-layer",
        "3",
        "--stage5-prototype-term-mode",
        "category_only",
        "--stage5-disable-prototype-proxy",
        "--stage5-margin-adv-threshold",
        "0.0",
        "--stage5-margin-adv-penalty",
        "0.05",
        "--stage6-max-instance-terms-per-category",
        "2",
        "--stage6-strict-synergy-threshold",
        "0.0",
        "--score-alpha",
        "256.0",
        "--candidate-overlap-penalty",
        "0.15",
        "--max-candidate-overlap",
        "0.8",
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--progress-every",
        str(args.progress_every),
        "--require-category-coverage",
    ]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fruit-focused real-category stage5/stage6 closure block")
    parser.add_argument("--models", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,Qwen/Qwen3-4B")
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--items-file", default=str(DEFAULT_ITEMS_FILE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(args)
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
