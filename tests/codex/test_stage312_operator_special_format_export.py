#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage312_operator_special_format_export import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["record_count"] > 0
    assert Path(summary["jsonl_path"]).exists()
    print("PASS")


if __name__ == "__main__":
    main()
