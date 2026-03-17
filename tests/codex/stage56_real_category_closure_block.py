from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "tests" / "codex" / "stage56_multimodel_sequential_pipeline.py"
DEFAULT_ITEMS_FILE = ROOT / "tests" / "codex" / "stage56_real_category_closure_items.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "tempdata" / "stage56_real_category_closure_block"


def load_real_category_items(items_file: Path) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    for line in items_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        term, category = [x.strip() for x in s.split(",", 1)]
        categories.setdefault(category, []).append(term)
    return categories


def validate_real_category_items_file(items_file: Path) -> Dict[str, List[str]]:
    categories = load_real_category_items(items_file)
    if not categories:
        raise ValueError(f"items file is empty: {items_file}")
    for category, terms in categories.items():
        if len(terms) != 3:
            raise ValueError(f"category {category} must contain exactly 3 terms, got {len(terms)}")
        if len(set(terms)) != len(terms):
            raise ValueError(f"category {category} contains duplicated terms")
        if category not in terms:
            raise ValueError(f"category {category} is missing its real category word")
        instance_terms = [term for term in terms if term != category]
        if len(instance_terms) != 2:
            raise ValueError(f"category {category} must contain exactly 2 instance terms")
    return categories


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
    ap = argparse.ArgumentParser(
        description="Run the real-category-word stage5/stage6 closure block sequentially for DeepSeek-7B and Qwen3-4B"
    )
    ap.add_argument("--models", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,Qwen/Qwen3-4B")
    ap.add_argument("--python-exe", default="")
    ap.add_argument("--items-file", default=str(DEFAULT_ITEMS_FILE))
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    validate_real_category_items_file(Path(args.items_file))
    command = build_command(args)
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
