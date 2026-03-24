#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage220_attention_closure_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["piece_count"] == 4
    assert summary["weakest_piece_name"] == "来源保真闭合"
    assert summary["strongest_piece_name"] == "前向携带来源"
    assert summary["top_gap_name"] == "天然来源保真不足"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
