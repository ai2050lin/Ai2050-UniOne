#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage303_shared_base_bias_cross_model_joint_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["review_score"] > 0.0
    assert len(summary["model_rows"]) == 2
    print("PASS")


if __name__ == "__main__":
    main()
