#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage335_local_operator_raw_cooccurrence_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage335_local_operator_raw_cooccurrence_map"
    assert summary["operator_count"] >= 1
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
