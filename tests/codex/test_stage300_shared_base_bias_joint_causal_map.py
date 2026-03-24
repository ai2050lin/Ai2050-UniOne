#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage300_shared_base_bias_joint_causal_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["joint_score"] > 0.0
    assert summary["joint_effect"] > 0.0
    print("PASS")


if __name__ == "__main__":
    main()
