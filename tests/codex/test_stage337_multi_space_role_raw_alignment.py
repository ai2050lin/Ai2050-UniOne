#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage337_multi_space_role_raw_alignment import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage337_multi_space_role_raw_alignment"
    assert 0.0 <= summary["alignment_score"] <= 2.0
    assert len(summary["alignment_rows"]) == 3
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
