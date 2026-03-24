#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage215_natural_trace_breakpoint_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["segment_count"] == 5
    assert summary["strongest_breakpoint_name"] == "天然保留段"
    assert summary["weakest_breakpoint_name"] == "修复迁移段"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
