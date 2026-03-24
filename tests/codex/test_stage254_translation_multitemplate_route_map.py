#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage254_translation_multitemplate_route_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["template_count"] >= 3
    assert summary["route_score"] > 0.25
    assert summary["strongest_template_name"] in {"weather", "apple", "window"}
    print("PASS")


if __name__ == "__main__":
    main()
