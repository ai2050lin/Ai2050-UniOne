#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage316_first_principles_eligibility_review import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["eligibility_score"] > 0.0
    assert "最小性" in summary["checklist"]
    print("PASS")


if __name__ == "__main__":
    main()
