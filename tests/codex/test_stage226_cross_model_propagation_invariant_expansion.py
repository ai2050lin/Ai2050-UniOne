#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage226_cross_model_propagation_invariant_expansion import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["weakest_piece_name"] == "过渡共同块"
    assert summary["strongest_piece_name"] == "稳定共同块"
    assert summary["top_gap_name"] == "跨模型传播不变量偏少"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
