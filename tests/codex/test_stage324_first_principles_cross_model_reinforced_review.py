#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage324_first_principles_cross_model_reinforced_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage324_first_principles_cross_model_reinforced_review"
    assert 0.0 <= summary["cross_model_score"] <= 1.0
    assert "跨模型稳定性" in summary["checklist"]
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
