#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage342_bias_deflection_3d_layer import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage342_bias_deflection_3d_layer"
    assert summary["layer_name"] == "bias_deflection_layer"
    assert len(summary["direction_nodes"]) >= 3
    assert len(summary["task_bias_rows"]) >= 2
    assert (OUTPUT_DIR / "summary.json").exists()
    assert (OUTPUT_DIR / "scene_layer.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
