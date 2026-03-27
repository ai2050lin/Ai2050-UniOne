#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage341_shared_carrier_3d_layer import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage341_shared_carrier_3d_layer"
    assert summary["layer_name"] == "shared_carrier_layer"
    assert len(summary["cluster_nodes"]) >= 4
    assert len(summary["task_bridges"]) >= 2
    assert (OUTPUT_DIR / "summary.json").exists()
    assert (OUTPUT_DIR / "scene_layer.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
