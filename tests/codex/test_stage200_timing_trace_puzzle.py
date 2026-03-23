#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage200_timing_trace_puzzle import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["weakest_piece_name"] == "retained_trace"
    assert summary["strongest_piece_name"] == "repair_trace"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
