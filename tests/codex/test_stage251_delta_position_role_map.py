#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage251_delta_position_role_map import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["dimension_count"] >= 20
    assert summary["brand_dim_count"] >= 10
    assert summary["role_counter"].get("水果内部差分", 0) > 0
    assert summary["role_counter"].get("动物内部差分", 0) > 0
    assert summary["role_counter"].get("品牌与跨类触发", 0) > 0
    print("PASS")


if __name__ == "__main__":
    main()
