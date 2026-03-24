#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage274_refactor_object_constraint_structure_bridge import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert len(summary["model_rows"]) == 2
    for row in summary["model_rows"]:
        assert row["bridge_score"] > 0.0
        assert len(row["bridge_rows"]) == 3
    print("PASS")


if __name__ == "__main__":
    main()
