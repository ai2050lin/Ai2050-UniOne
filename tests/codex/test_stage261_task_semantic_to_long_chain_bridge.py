#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage261_task_semantic_to_long_chain_bridge import run_analysis


def main() -> None:
    summary = run_analysis()
    assert summary["piece_count"] == 4
    assert 0.0 <= summary["bridge_score"] <= 1.0
    output_dir = Path("tests/codex_temp/stage261_task_semantic_to_long_chain_bridge_20260324")
    assert (output_dir / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
