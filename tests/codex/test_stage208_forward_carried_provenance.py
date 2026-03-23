#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage208_forward_carried_provenance import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["top_gap_name"] == "天然保留不足"
    assert summary["forward_carried_score"] < 0.5
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
