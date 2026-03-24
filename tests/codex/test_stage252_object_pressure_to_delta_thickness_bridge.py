#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage252_object_pressure_to_delta_thickness_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["word_count"] >= 16
    assert summary["bridge_score"] >= 0.5
    assert summary["strongest_word_name"] != summary["weakest_word_name"]
    print("PASS")


if __name__ == "__main__":
    main()
