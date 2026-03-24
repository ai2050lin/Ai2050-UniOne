#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage255_translation_token_role_refinement import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=False)
    assert summary["variant_count"] >= 4
    assert summary["strongest_role_name"] in {
        "remove_translate_word",
        "remove_target_word",
        "remove_source_word",
        "remove_output_constraint",
    }
    print("PASS")


if __name__ == "__main__":
    main()
