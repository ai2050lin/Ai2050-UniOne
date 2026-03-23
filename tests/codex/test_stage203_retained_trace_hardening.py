#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage203_retained_trace_hardening import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["weakest_piece_name"] == "天然保留"
    assert summary["strongest_piece_name"] == "修复迁移"
    assert summary["hardening_gain_space"] > 0.7
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
