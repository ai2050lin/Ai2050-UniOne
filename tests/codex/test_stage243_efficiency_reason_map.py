#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage243_efficiency_reason_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["reason_score"] > 0.7
    assert summary["strongest_reason_name"] in {"共享复用强度", "路径放大强度", "局部差分稀疏性", "候选结构一致性"}
    print("PASS")


if __name__ == "__main__":
    main()
