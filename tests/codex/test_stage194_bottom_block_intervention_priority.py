#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage194_bottom_block_intervention_priority import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["target_count"] == 17
    assert summary["top_priority_name"] == "回收束"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
