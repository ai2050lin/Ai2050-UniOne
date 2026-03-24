#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage325_shared_carrier_cross_task_core_compression import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage325_shared_carrier_cross_task_core_compression"
    assert 0.0 <= summary["compression_score"] <= 1.0
    assert len(summary["compressed_rows"]) == 1
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
