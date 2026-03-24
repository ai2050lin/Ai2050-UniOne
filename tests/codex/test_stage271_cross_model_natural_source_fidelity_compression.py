#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage271_cross_model_natural_source_fidelity_compression import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    assert summary["common_repair_fidelity_score"] >= summary["common_natural_fidelity_score"]
    for row in summary["model_rows"]:
        assert len(row["probe_rows"]) == 3
    print("PASS")


if __name__ == "__main__":
    main()
