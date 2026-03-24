#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage311_bias_deflection_operator_core_compression import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["compression_score"] > 0.0
    assert summary["core_count"] > 0
    print("PASS")


if __name__ == "__main__":
    main()
