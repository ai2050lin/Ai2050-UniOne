#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage231_source_fidelity_parameter_structure_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["top_gap_name"] == "天然来源保真参数支持仍然不足"
    assert summary["weakest_piece_name"] == "天然来源保真参数支持"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
