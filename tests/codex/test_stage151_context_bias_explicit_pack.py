#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage151_context_bias_explicit_pack import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["target_variable"] == "b"
    assert summary["case_count"] == 320
    assert summary["family_count"] == 5
    assert summary["difficulty_count"] == 4
    assert summary["control_type_count"] == 4
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
