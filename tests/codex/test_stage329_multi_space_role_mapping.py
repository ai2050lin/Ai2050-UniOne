#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage329_multi_space_role_mapping import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage329_multi_space_role_mapping"
    assert 0.0 <= summary["mapping_score"] <= 2.0
    assert len(summary["space_rows"]) == 6
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
