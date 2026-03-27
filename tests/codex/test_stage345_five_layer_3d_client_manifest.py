#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage345_five_layer_3d_client_manifest import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage345_five_layer_3d_client_manifest"
    assert len(summary["layers"]) == 5
    assert (OUTPUT_DIR / "summary.json").exists()
    assert (OUTPUT_DIR / "agi_3d_client_scene_v1.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
