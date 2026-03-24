#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage252_translation_component_role_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=False)
    assert summary["component_count"] >= 3
    assert summary["strongest_component_name"] in {"remove_translate", "remove_target", "remove_source"}
    print("PASS")


if __name__ == "__main__":
    main()
