#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage328_cross_model_common_core_compression import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage328_cross_model_common_core_compression"
    assert 0.0 <= summary["common_core_score"] <= 1.0
    assert len(summary["common_rows"]) == 3
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
