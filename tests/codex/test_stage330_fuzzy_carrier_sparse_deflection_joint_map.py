#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage330_fuzzy_carrier_sparse_deflection_joint_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage330_fuzzy_carrier_sparse_deflection_joint_map"
    assert 0.0 <= summary["joint_score"] <= 2.0
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
