#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage331_layerwise_relay_independent_core_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage331_layerwise_relay_independent_core_map"
    assert 0.0 <= summary["relay_score"] <= 2.0
    assert len(summary["layer_rows"]) == 3
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
