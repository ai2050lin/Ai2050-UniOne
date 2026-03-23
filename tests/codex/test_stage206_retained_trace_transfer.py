#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage206_retained_trace_transfer import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["weakest_piece_name"] == "天然保留"
    assert summary["strongest_piece_name"] == "修复迁移"
    assert summary["transfer_score"] < 0.5
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
