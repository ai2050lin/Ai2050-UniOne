#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage296_base_fixed_bias_swap_experiment import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["experiment_score"] > 0.0
    assert len(summary["experiment_rows"]) > 0
    print("PASS")


if __name__ == "__main__":
    main()
