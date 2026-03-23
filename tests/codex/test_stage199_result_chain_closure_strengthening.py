#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage199_result_chain_closure_strengthening import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["result_chain_priority"] == "一级干预"
    assert summary["closure_gap"] > 0.5
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
