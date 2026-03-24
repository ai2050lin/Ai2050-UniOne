#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage269_parameter_position_causal_intervention_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    for row in summary["model_rows"]:
        assert row["parameter_intervention_score"] > 0.0
        assert row["contrast_count"] >= 3
        assert len(row["hot_dim_rows"]) > 0
    print("PASS")


if __name__ == "__main__":
    main()
