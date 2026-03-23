#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage188_apple_neuron_role_card import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["role_count"] == 7
    assert summary["strongest_role_name"] == "路径束"
    assert summary["weakest_role_name"] == "回收束"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
