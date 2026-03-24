#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage240_structure_efficiency_candidate_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["best_candidate_name"] == "共享底盘 + 局部差分 + 路径放大"
    assert summary["worst_candidate_name"] == "局部差分单独结构"
    print("PASS")


if __name__ == "__main__":
    main()
