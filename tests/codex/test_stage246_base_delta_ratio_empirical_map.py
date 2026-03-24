#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage246_base_delta_ratio_empirical_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["best_ratio_name"] in {"95/5", "90/10"}
    assert summary["best_ratio_name"] != summary["worst_ratio_name"]
    print("PASS")


if __name__ == "__main__":
    main()
