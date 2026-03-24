#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage293_base_to_bias_operation_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["bridge_score"] > 0.0
    print("PASS")


if __name__ == "__main__":
    main()
