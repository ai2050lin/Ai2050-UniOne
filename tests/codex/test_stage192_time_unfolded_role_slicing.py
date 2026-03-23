#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage192_time_unfolded_role_slicing import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["slice_count"] == 5
    assert summary["strongest_slice_name"] == "中段选路切片"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
