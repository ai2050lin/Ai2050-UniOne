#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage237_noun_parameter_efficiency_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["top_gap_name"] == "差分分裂效率仍然低于共享复用效率"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
