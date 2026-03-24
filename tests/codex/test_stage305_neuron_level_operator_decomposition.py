#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage305_neuron_level_operator_decomposition import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["decomposition_score"] > 0.0
    assert summary["operator_count"] == 3
    print("PASS")


if __name__ == "__main__":
    main()
