#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage141_cross_model_layer_isomorphism import run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["status_short"] == "cross_model_layer_isomorphism_ready"
    assert summary["model_count"] == 3
    assert summary["pair_count"] == 2
    assert 0.0 <= summary["mean_layer_isomorphism_score"] <= 1.0
    assert Path("tests/codex_temp/stage141_cross_model_layer_isomorphism_20260323/summary.json").exists()
    assert Path("tests/codex_temp/stage141_cross_model_layer_isomorphism_20260323/STAGE141_CROSS_MODEL_LAYER_ISOMORPHISM_REPORT.md").exists()
    print("PASS")


if __name__ == "__main__":
    main()
