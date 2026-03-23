#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage205_reentry_closure_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["weakest_piece_name"] == "时序痕迹桥"
    assert summary["closure_gap"] > 0.5
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
