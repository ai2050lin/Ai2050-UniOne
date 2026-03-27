#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage343_layerwise_amplification_3d_layer import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage343_layerwise_amplification_3d_layer"
    assert summary["layer_name"] == "layerwise_amplification_layer"
    assert len(summary["relay_bands"]) == 3
    assert (OUTPUT_DIR / "summary.json").exists()
    assert (OUTPUT_DIR / "scene_layer.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
