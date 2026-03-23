#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from stage153_noun_verb_result_hardening_pack import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["case_count"] == 384
    assert summary["family_count"] == 6
    assert summary["fruit_seed_count"] >= 6
    assert set(summary["target_variables"]) >= {"a", "g", "f"}
    assert Path(OUTPUT_DIR / "summary.json").exists()
    print("PASS")


if __name__ == "__main__":
    main()
