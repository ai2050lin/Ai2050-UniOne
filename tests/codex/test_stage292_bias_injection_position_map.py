#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage292_bias_injection_position_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["bias_score"] > 0.0
    assert summary["fruit_bias_count"] > 0
    print("PASS")


if __name__ == "__main__":
    main()
