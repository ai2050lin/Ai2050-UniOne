#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage202_reentry_closure_puzzle import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["top_gap_name"] == "来源痕迹天然保留"
    assert summary["reentry_closure_score"] < 0.5
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
