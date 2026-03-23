#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage201_phase_route_puzzle import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["dominant_band_name"] == "early"
    assert summary["phase_route_score"] > 0.45
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
