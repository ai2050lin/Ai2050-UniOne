#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stage245_large_scale_noun_shared_delta_tensor import OUTPUT_DIR, run_analysis


def main() -> None:
    summary = run_analysis(output_dir=OUTPUT_DIR, force=True)
    assert summary["noun_count"] > 20000
    assert summary["sparse_delta_mean"] > summary["hard_delta_mean"]
    assert summary["shared_base_mean"] > summary["local_delta_mean"]
    print("PASS")


if __name__ == "__main__":
    main()
