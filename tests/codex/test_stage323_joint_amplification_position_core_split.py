#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage323_joint_amplification_position_core_split import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(force=True)
    assert summary["experiment_id"] == "stage323_joint_amplification_position_core_split"
    assert 0.0 <= summary["position_split_score"] <= 1.0
    assert len(summary["position_rows"]) == 3
    assert (OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
