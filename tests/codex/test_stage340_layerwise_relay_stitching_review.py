#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage340_layerwise_relay_stitching_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage340_layerwise_relay_stitching_review"
    assert 0.0 <= summary["review_score"] <= 2.0
    assert len(summary["layer_rows"]) == 3
    assert len(summary["stitch_rows"]) == 4
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
