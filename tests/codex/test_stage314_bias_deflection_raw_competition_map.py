#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage314_bias_deflection_raw_competition_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["raw_competition_score"] > 0.0
    print("PASS")


if __name__ == "__main__":
    main()
