#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage236_apple_pear_parameter_delta import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 7
    assert summary["top_gap_name"] == "苹果与梨子的差分边界仍然偏薄"
    assert summary["apple_banana_similarity"] > summary["apple_pear_similarity"]
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
