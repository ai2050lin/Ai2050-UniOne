#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage334_bias_deflection_direction_raw_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage334_bias_deflection_direction_raw_map"
    assert summary["direction_count"] >= 5
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
