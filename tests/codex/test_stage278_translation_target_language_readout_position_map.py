#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage278_translation_target_language_readout_position_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    for row in summary["model_rows"]:
        assert len(row["english_readout_dim_rows"]) > 0
        assert len(row["chinese_readout_dim_rows"]) > 0
    print("PASS")


if __name__ == "__main__":
    main()
