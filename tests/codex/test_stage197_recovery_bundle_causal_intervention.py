#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from stage197_recovery_bundle_causal_intervention import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["top_target_name"] == "回收束"
    assert summary["second_target_name"] == "原生绑定"
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
