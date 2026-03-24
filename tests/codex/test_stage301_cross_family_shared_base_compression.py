#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage301_cross_family_shared_base_compression import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["compression_score"] > 0.0
    print("PASS")


if __name__ == "__main__":
    main()
