#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage299_bias_position_role_card import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["role_score"] > 0.0
    assert len(summary["position_rows"]) > 0
    print("PASS")


if __name__ == "__main__":
    main()
