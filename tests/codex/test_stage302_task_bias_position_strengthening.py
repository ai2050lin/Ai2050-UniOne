#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage302_task_bias_position_strengthening import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["strengthening_score"] > 0.0
    assert len(summary["task_rows"]) == 2
    print("PASS")


if __name__ == "__main__":
    main()
