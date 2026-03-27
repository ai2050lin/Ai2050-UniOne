#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage338_fuzzy_carrier_cluster_stability_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage338_fuzzy_carrier_cluster_stability_review"
    assert 0.0 <= summary["review_score"] <= 2.0
    assert len(summary["stability_rows"]) >= 4
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
