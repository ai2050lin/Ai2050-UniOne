#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage154_apple_fruit_shared_core import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["target_word"] == "apple"
    assert summary["fruit_member_count"] > 100
    assert len(summary["top_fruit_neighbors"]) == 8
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
