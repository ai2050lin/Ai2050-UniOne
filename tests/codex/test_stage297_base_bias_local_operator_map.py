#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage297_base_bias_local_operator_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["operator_score"] > 0.0
    assert summary["operator_count"] >= 3
    print("PASS")


if __name__ == "__main__":
    main()
