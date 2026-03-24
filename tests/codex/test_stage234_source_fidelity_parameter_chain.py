#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage234_source_fidelity_parameter_chain import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 5
    assert summary["top_gap_name"] == "天然来源保真参数起点仍然过弱"
    assert summary["weakest_piece_name"] == "天然来源保真参数起点"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
