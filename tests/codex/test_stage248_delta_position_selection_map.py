#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage248_delta_position_selection_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["word_count"] >= 24
    assert summary["top_dim_delta_mean"] > summary["global_delta_mean"]
    assert summary["low_base_high_delta_ratio"] >= 0.15
    print("PASS")


if __name__ == "__main__":
    main()
