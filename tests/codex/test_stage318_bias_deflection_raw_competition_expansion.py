#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage318_bias_deflection_raw_competition_expansion import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage318_bias_deflection_raw_competition_expansion"
    assert 0.0 <= summary["raw_expansion_score"] <= 1.0
    assert len(summary["competition_axes"]) >= 4
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
