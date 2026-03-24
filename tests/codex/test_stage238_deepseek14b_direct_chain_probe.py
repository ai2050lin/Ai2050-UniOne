#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage238_deepseek14b_direct_chain_probe import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["probe_count"] == 4
    assert summary["model_name"] == "deepseek-r1:14b"
    assert summary["correct_count"] >= 2
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
