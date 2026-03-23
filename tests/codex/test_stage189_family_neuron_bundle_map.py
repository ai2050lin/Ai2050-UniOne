#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage189_family_neuron_bundle_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["bundle_count"] == 5
    assert summary["strongest_bundle_name"] == "tool"
    assert summary["weakest_bundle_name"] == "tool"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
