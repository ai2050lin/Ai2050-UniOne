#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage260_full_semantic_role_total_map import run_analysis


def main() -> None:
    summary = run_analysis()
    assert summary["role_count"] == 6
    assert 0.0 <= summary["total_score"] <= 1.0
    output_dir = Path("tests/codex_temp/stage260_full_semantic_role_total_map_20260324")
    assert (output_dir / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
