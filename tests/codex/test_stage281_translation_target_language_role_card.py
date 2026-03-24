#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage281_translation_target_language_role_card import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    for row in summary["model_rows"]:
        assert row["role_score"] > 0.0
    print("PASS")


if __name__ == "__main__":
    main()
