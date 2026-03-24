#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage320_first_principles_eligibility_reinforced_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage320_first_principles_eligibility_reinforced_review"
    assert 0.0 <= summary["reinforced_score"] <= 1.0
    assert "最小性" in summary["checklist"]
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
