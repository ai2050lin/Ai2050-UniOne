from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_real_category_closure_block import (  # noqa: E402
    DEFAULT_ITEMS_FILE,
    DEFAULT_OUTPUT_ROOT,
    build_command,
)


def make_args(**overrides):
    defaults = {
        "models": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,Qwen/Qwen3-4B",
        "python_exe": "python",
        "items_file": str(DEFAULT_ITEMS_FILE),
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "dtype": "bfloat16",
        "device": "cuda",
        "seed": 42,
        "progress_every": 10,
        "resume": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_command_uses_real_category_items_file():
    args = make_args()
    command = build_command(args)
    assert "--items-file" in command
    assert str(Path(args.items_file)) in command
    assert "--require-category-coverage" in command


def test_build_command_supports_resume_and_dry_run():
    args = make_args(resume=True, dry_run=True)
    command = build_command(args)
    assert "--resume" in command
    assert "--dry-run" in command


def test_real_category_items_file_has_four_categories_and_category_words():
    rows = []
    for line in DEFAULT_ITEMS_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        term, category = [x.strip() for x in s.split(",", 1)]
        rows.append((term, category))
    categories = {}
    for term, category in rows:
        categories.setdefault(category, []).append(term)
    assert sorted(categories) == ["animal", "human", "tech", "vehicle"]
    assert all(len(terms) == 3 for terms in categories.values())
    assert all(category in terms for category, terms in categories.items())
