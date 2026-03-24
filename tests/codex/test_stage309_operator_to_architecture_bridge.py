#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage309_operator_to_architecture_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["bridge_score"] > 0.0
    assert len(summary["architecture_template"]) == 3
    print("PASS")


if __name__ == "__main__":
    main()
