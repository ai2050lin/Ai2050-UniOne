#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage284_multihop_translation_command_probe import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    for row in summary["model_rows"]:
        assert len(row["prompt_rows"]) == 4
    print("PASS")


if __name__ == "__main__":
    main()
