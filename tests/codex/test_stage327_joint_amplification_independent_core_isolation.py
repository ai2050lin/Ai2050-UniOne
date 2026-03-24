#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage327_joint_amplification_independent_core_isolation import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage327_joint_amplification_independent_core_isolation"
    assert 0.0 <= summary["isolation_score"] <= 1.0
    assert len(summary["isolated_rows"]) == 3
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
