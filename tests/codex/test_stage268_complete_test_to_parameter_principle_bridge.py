#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage268_complete_test_to_parameter_principle_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    assert summary["score_gap"] >= 0.0
    print("PASS")


if __name__ == "__main__":
    main()
