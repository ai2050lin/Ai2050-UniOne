#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage198_provenance_trace_continuity_tracking import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["continuity_gap"] > 0.7
    assert summary["priority_level"] == "二级干预"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
