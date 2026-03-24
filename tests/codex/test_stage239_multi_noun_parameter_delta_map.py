#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage239_multi_noun_parameter_delta_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["map_score"] > 0.3
    assert summary["strongest_piece_name"] != summary["weakest_piece_name"]
    print("PASS")


if __name__ == "__main__":
    main()
