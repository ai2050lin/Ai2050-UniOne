#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage242_large_scale_noun_shared_delta_matrix import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["noun_count"] > 20000
    assert summary["shared_base_strength"] > summary["local_delta_strength"]
    print("PASS")


if __name__ == "__main__":
    main()
