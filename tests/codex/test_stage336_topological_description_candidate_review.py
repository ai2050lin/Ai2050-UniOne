#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage336_topological_description_candidate_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage336_topological_description_candidate_review"
    assert 0.0 <= summary["candidate_score"] <= 2.0
    assert len(summary["candidate_rows"]) == 4
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
