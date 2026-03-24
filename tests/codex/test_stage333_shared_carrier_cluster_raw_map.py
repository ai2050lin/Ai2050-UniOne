#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage333_shared_carrier_cluster_raw_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage333_shared_carrier_cluster_raw_map"
    assert summary["cluster_count"] >= 4
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
