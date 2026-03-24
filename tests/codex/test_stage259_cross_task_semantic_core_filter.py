#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage259_cross_task_semantic_core_filter import run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["core_count"] == 4
    assert 0.0 <= summary["filter_score"] <= 1.0
    assert 0 <= summary["stable_core_count"] <= summary["core_count"]
    output_dir = Path("tests/codex_temp/stage259_cross_task_semantic_core_filter_20260324")
    assert (output_dir / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
