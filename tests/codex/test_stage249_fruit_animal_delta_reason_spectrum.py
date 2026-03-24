#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage249_fruit_animal_delta_reason_spectrum import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["word_count"] >= 16
    assert summary["fruit_mean_family_share"] > 0.3
    assert summary["animal_mean_margin"] > summary["fruit_mean_margin"]
    print("PASS")


if __name__ == "__main__":
    main()
