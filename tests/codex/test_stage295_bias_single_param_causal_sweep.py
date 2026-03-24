#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage295_bias_single_param_causal_sweep import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["causal_score"] > 0.0
    assert summary["candidate_count"] > 0
    print("PASS")


if __name__ == "__main__":
    main()
