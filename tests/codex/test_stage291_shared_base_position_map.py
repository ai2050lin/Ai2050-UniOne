#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage291_shared_base_position_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["shared_base_score"] > 0.0
    assert summary["base_candidate_count"] > 0
    print("PASS")


if __name__ == "__main__":
    main()
